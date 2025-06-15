import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import os
import difflib


# 配置参数
class Config:
    def __init__(self):
        self.model_name = 'bert-base-chinese'
        self.max_len = 128
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.epochs = 10
        self.num_target_groups = 5  # 5类目标群体
        self.tag2id = {
            'O': 0,
            'B-TARGET': 1,
            'I-TARGET': 2,
            'B-ARG': 3,
            'I-ARG': 4
        }
        self.id2tag = {v: k for k, v in self.tag2id.items()}
        self.target_group_labels = ['Sexism', 'Racism', 'Region', 'LGBTQ', 'others', 'non-hate']
        self.hateful_labels = ['hate', 'non-hate']
        self.ner_loss_weight = 2.0  # 增加NER损失权重


config = Config()


# 数据加载与预处理
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_output(output_str):
    """解析output字符串为四元组列表"""
    quads = []
    # 分割多个四元组
    for quad_str in output_str.split('[SEP]'):
        quad_str = quad_str.strip()
        if not quad_str:
            continue
        # 移除结尾的[END]
        if quad_str.endswith('[END]'):
            quad_str = quad_str[:-5].strip()
        parts = quad_str.split(' | ')
        if len(parts) == 4:
            quads.append(tuple(parts))
    return quads


def find_spans(text, phrase):
    """在文本中查找短语的精确位置"""
    if phrase == 'NULL':
        return []
    pattern = re.compile(re.escape(phrase))
    matches = pattern.finditer(text)
    return [(match.start(), match.end()) for match in matches]


class HateSpeechDataset(Dataset):
    def __init__(self, data, tokenizer, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['content']

        # 使用快速分词器处理文本
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=config.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        # 提取并处理结果
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()

        # 训练数据需要标签
        if self.is_train:
            quads = parse_output(item['output'])

            # 初始化标签
            ner_tags = [config.tag2id['O']] * config.max_len
            target_group_labels = torch.zeros(len(config.target_group_labels), dtype=torch.float)
            hateful_label = torch.tensor(0, dtype=torch.long)  # 默认non-hate

            # 处理每个四元组
            for quad in quads:
                target, arg, target_group, hateful = quad

                # 标记Target和Argument
                for phrase, tag_type in [(target, 'TARGET'), (arg, 'ARG')]:
                    spans = find_spans(text, phrase)
                    for start, end in spans:
                        # 查找在tokenized文本中的位置
                        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                            if char_start == 0 and char_end == 0:  # 特殊token
                                continue
                            # 更精确的匹配逻辑
                            if start <= char_start < end or start < char_end <= end or (
                                    char_start <= start and end <= char_end):
                                prefix = 'B-' if char_start == start else 'I-'
                                ner_tags[token_idx] = config.tag2id[prefix + tag_type]

                # 处理目标群体标签
                if target_group != 'non-hate':
                    for group in target_group.split(', '):
                        if group in config.target_group_labels:
                            group_idx = config.target_group_labels.index(group)
                            target_group_labels[group_idx] = 1

                # 处理仇恨标签
                if hateful == 'hate':
                    hateful_label = torch.tensor(1, dtype=torch.long)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'ner_tags': torch.tensor(ner_tags, dtype=torch.long),
                'target_group_labels': target_group_labels,
                'hateful_label': hateful_label,
                'offset_mapping': offset_mapping,
                'text': text
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'offset_mapping': offset_mapping,
                'text': text,
                'id': item['id']
            }


# 改进的模型架构
class HateSpeechModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = torch.nn.Dropout(0.1)

        # NER任务头 - 增加隐藏层大小
        self.ner_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, len(config.tag2id)))

        # 目标群体分类头
        self.target_group_classifier = torch.nn.Linear(self.bert.config.hidden_size, len(config.target_group_labels))

        # 仇恨分类头
        self.hate_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size + len(config.target_group_labels), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # NER任务
        ner_logits = self.ner_classifier(self.dropout(sequence_output))

        # 目标群体分类（使用[CLS]标记）
        pooled_output = sequence_output[:, 0, :]
        target_group_logits = self.target_group_classifier(self.dropout(pooled_output))

        # 仇恨分类 - 结合CLS表示和目标群体信息
        target_group_probs = torch.sigmoid(target_group_logits)
        hate_input = torch.cat([pooled_output, target_group_probs], dim=1)
        hate_logits = self.hate_classifier(self.dropout(hate_input))

        return ner_logits, target_group_logits, hate_logits


# 计算字符串相似度
def string_similarity(a, b):
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# 评估四元组准确率
def evaluate_quad_accuracy(model, data_loader, device, tokenizer):
    model.eval()

    # 初始化统计指标
    total_samples = 0
    target_match = 0
    argument_match = 0
    target_group_match = 0
    hateful_match = 0
    quad_hard_match = 0
    quad_soft_match = 0

    # 新增指标：实体位置准确率
    target_position_acc = 0
    argument_position_acc = 0

    # 新增指标：实体边界准确率
    target_boundary_acc = 0
    argument_boundary_acc = 0

    with torch.no_grad():
        for batch in data_loader:
            texts = batch['text']
            true_outputs = batch.get('output', [])

            for i, text in enumerate(texts):
                # 预测四元组
                pred_output = predict_quads(model, tokenizer, text, device)

                # 解析预测的四元组
                pred_quads = parse_output(pred_output)
                if not pred_quads:
                    pred_quads = [('NULL', 'NULL', 'non-hate', 'non-hate')]

                # 解析真实四元组
                if i < len(true_outputs):
                    true_quads = parse_output(true_outputs[i])
                else:
                    true_quads = [('NULL', 'NULL', 'non-hate', 'non-hate')]

                if not true_quads:
                    true_quads = [('NULL', 'NULL', 'non-hate', 'non-hate')]

                # 只比较第一个四元组
                pred_quad = pred_quads[0]
                true_quad = true_quads[0]

                # 计算各个元素的匹配情况
                # 1. Target匹配
                target_sim = string_similarity(pred_quad[0], true_quad[0])
                if target_sim >= 0.5:
                    target_match += 1

                # 2. Argument匹配
                arg_sim = string_similarity(pred_quad[1], true_quad[1])
                if arg_sim >= 0.5:
                    argument_match += 1

                # 3. Target Group匹配
                pred_groups = set(pred_quad[2].split(', '))
                true_groups = set(true_quad[2].split(', '))
                if pred_groups == true_groups:
                    target_group_match += 1

                # 4. Hateful匹配
                if pred_quad[3] == true_quad[3]:
                    hateful_match += 1

                # 5. 整个四元组硬匹配
                if pred_quad == true_quad:
                    quad_hard_match += 1

                # 6. 整个四元组软匹配
                if (target_sim >= 0.5 and arg_sim >= 0.5 and
                        pred_groups == true_groups and
                        pred_quad[3] == true_quad[3]):
                    quad_soft_match += 1

                # 7. 实体位置准确率（新增）
                if pred_quad[0] != 'NULL' and true_quad[0] != 'NULL':
                    pred_target_span = find_spans(text, pred_quad[0])
                    true_target_span = find_spans(text, true_quad[0])
                    if pred_target_span and true_target_span:
                        pred_start, pred_end = pred_target_span[0]
                        true_start, true_end = true_target_span[0]
                        if abs(pred_start - true_start) <= 2 and abs(pred_end - true_end) <= 2:
                            target_position_acc += 1

                if pred_quad[1] != 'NULL' and true_quad[1] != 'NULL':
                    pred_arg_span = find_spans(text, pred_quad[1])
                    true_arg_span = find_spans(text, true_quad[1])
                    if pred_arg_span and true_arg_span:
                        pred_start, pred_end = pred_arg_span[0]
                        true_start, true_end = true_arg_span[0]
                        if abs(pred_start - true_start) <= 2 and abs(pred_end - true_end) <= 2:
                            argument_position_acc += 1

                # 8. 实体边界准确率（新增）
                if pred_quad[0] != 'NULL' and true_quad[0] != 'NULL':
                    if pred_quad[0] in true_quad[0] or true_quad[0] in pred_quad[0]:
                        target_boundary_acc += 1

                if pred_quad[1] != 'NULL' and true_quad[1] != 'NULL':
                    if pred_quad[1] in true_quad[1] or true_quad[1] in pred_quad[1]:
                        argument_boundary_acc += 1

                total_samples += 1

    # 计算准确率
    metrics = {
        'target_acc': target_match / total_samples if total_samples > 0 else 0,
        'argument_acc': argument_match / total_samples if total_samples > 0 else 0,
        'target_group_acc': target_group_match / total_samples if total_samples > 0 else 0,
        'hateful_acc': hateful_match / total_samples if total_samples > 0 else 0,
        'quad_hard_acc': quad_hard_match / total_samples if total_samples > 0 else 0,
        'quad_soft_acc': quad_soft_match / total_samples if total_samples > 0 else 0,
        'target_position_acc': target_position_acc / total_samples if total_samples > 0 else 0,
        'argument_position_acc': argument_position_acc / total_samples if total_samples > 0 else 0,
        'target_boundary_acc': target_boundary_acc / total_samples if total_samples > 0 else 0,
        'argument_boundary_acc': argument_boundary_acc / total_samples if total_samples > 0 else 0,
    }

    return metrics


# 训练函数
def train_model(model, train_loader, val_loader, device, tokenizer):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    model.to(device)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        # 初始化损失计数器
        ner_loss_total = 0
        group_loss_total = 0
        hate_loss_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ner_tags = batch['ner_tags'].to(device)
            target_group_labels = batch['target_group_labels'].to(device)
            hateful_label = batch['hateful_label'].to(device)

            optimizer.zero_grad()

            # 前向传播
            ner_logits, target_group_logits, hate_logits = model(
                input_ids, attention_mask
            )

            # 计算损失 - 增加NER损失权重
            loss_ner = torch.nn.CrossEntropyLoss(ignore_index=0)(
                ner_logits.view(-1, len(config.tag2id)),
                ner_tags.view(-1)
            ) * config.ner_loss_weight

            loss_target_group = torch.nn.BCEWithLogitsLoss()(
                target_group_logits, target_group_labels
            )

            # 仇恨标签损失
            hate_target = hateful_label.float().view(-1, 1)
            hate_loss = torch.nn.BCELoss()(
                hate_logits,
                hate_target
            )

            total_batch_loss = loss_ner + loss_target_group + hate_loss
            total_loss += total_batch_loss.item()
            batch_count += 1

            # 记录各项损失
            ner_loss_total += loss_ner.item()
            group_loss_total += loss_target_group.item()
            hate_loss_total += hate_loss.item()

            # 反向传播
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # 打印平均损失
        avg_loss = total_loss / batch_count
        avg_ner_loss = ner_loss_total / batch_count
        avg_group_loss = group_loss_total / batch_count
        avg_hate_loss = hate_loss_total / batch_count

        print(f"Epoch {epoch + 1} - Losses:")
        print(
            f"  Total: {avg_loss:.4f}, NER: {avg_ner_loss:.4f}, Group: {avg_group_loss:.4f}, Hate: {avg_hate_loss:.4f}")

        # 验证 - 计算F1分数
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1} - Val F1: {val_metrics['f1']:.4f}")

        # 验证 - 计算四元组准确率
        quad_metrics = evaluate_quad_accuracy(model, val_loader, device, tokenizer)
        print(f"Epoch {epoch + 1} - Quad Accuracy Metrics:")
        print(f"  Target Acc: {quad_metrics['target_acc']:.4f}")
        print(f"  Argument Acc: {quad_metrics['argument_acc']:.4f}")
        print(f"  Target Group Acc: {quad_metrics['target_group_acc']:.4f}")
        print(f"  Hateful Acc: {quad_metrics['hateful_acc']:.4f}")
        print(f"  Quad Hard Acc: {quad_metrics['quad_hard_acc']:.4f}")
        print(f"  Quad Soft Acc: {quad_metrics['quad_soft_acc']:.4f}")
        print(f"  Target Position Acc: {quad_metrics['target_position_acc']:.4f}")
        print(f"  Argument Position Acc: {quad_metrics['argument_position_acc']:.4f}")
        print(f"  Target Boundary Acc: {quad_metrics['target_boundary_acc']:.4f}")
        print(f"  Argument Boundary Acc: {quad_metrics['argument_boundary_acc']:.4f}")

        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pt')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pt'))
    return model


# 评估函数
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hateful_label = batch['hateful_label'].to(device)

            # 前向传播
            _, _, hate_logits = model(input_ids, attention_mask)

            # 获取预测
            hate_preds = (hate_logits > 0.5).long().squeeze()
            all_preds.extend(hate_preds.cpu().numpy())
            all_labels.extend(hateful_label.cpu().numpy())

    # 计算F1分数
    f1 = f1_score(all_labels, all_preds, average='macro')
    return {'f1': f1}


# 改进的实体抽取逻辑
def extract_entities(ner_preds, offset_mapping, text):
    """改进的实体抽取函数"""
    entities = {'TARGET': [], 'ARG': []}
    current_entity = None
    current_start = None
    current_end = None
    current_text = ""

    for i, (pred, offset) in enumerate(zip(ner_preds, offset_mapping)):
        if offset[0] == 0 and offset[1] == 0:  # 特殊token
            continue

        tag = config.id2tag[pred]
        if tag.startswith('B-'):
            # 保存之前的实体
            if current_entity is not None:
                entities[current_entity].append((current_start, current_end, current_text))

            # 开始新实体
            current_entity = tag[2:]
            current_start = offset[0]
            current_end = offset[1]
            current_text = text[offset[0]:offset[1]]

        elif tag.startswith('I-') and current_entity == tag[2:]:
            # 继续当前实体
            current_end = offset[1]
            current_text = text[current_start:current_end]

        else:
            # 结束当前实体
            if current_entity is not None:
                entities[current_entity].append((current_start, current_end, current_text))
                current_entity = None

    # 保存最后一个实体
    if current_entity is not None:
        entities[current_entity].append((current_start, current_end, current_text))

    return entities


# 改进的预测函数
def predict_quads(model, tokenizer, text, device):
    """预测文本的四元组 - 改进版本"""
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config.max_len,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'][0].tolist()

    with torch.no_grad():
        ner_logits, target_group_logits, hate_logits = model(input_ids, attention_mask)

    # 处理NER预测
    ner_preds = torch.argmax(ner_logits, dim=2).squeeze(0).cpu().numpy()

    # 改进的实体抽取
    entities = extract_entities(ner_preds, offset_mapping, text)

    # 提取目标群体和仇恨标签
    target_group_probs = torch.sigmoid(target_group_logits).squeeze(0).cpu().numpy()
    hate_prob = hate_logits.squeeze().item()  # 仇恨概率

    # 确定目标群体
    target_groups = []
    for i, prob in enumerate(target_group_probs):
        if prob > 0.5 and i < len(config.target_group_labels):
            target_groups.append(config.target_group_labels[i])

    # 确定仇恨标签
    hate_label = 'hate' if hate_prob > 0.5 else 'non-hate'

    # 确保标签一致性
    if hate_label == 'non-hate':
        target_groups = ['non-hate']
    elif hate_label == 'hate' and not any(group != 'non-hate' for group in target_groups):
        target_groups = ['others']

    # 构建四元组 - 改进实体选择
    quads = []

    # 选择最相关的Target - 优先选择完整的实体
    target_entities = entities.get('TARGET', [])
    if target_entities:
        # 优先选择完整的单词实体
        complete_entities = [e for e in target_entities if e[2].strip() and not e[2].endswith(('，', '。', '？', '！'))]

        if complete_entities:
            # 选择最长的完整实体
            target_entities = sorted(complete_entities, key=lambda x: len(x[2]), reverse=True)
            target_span = target_entities[0]
            target_text = target_span[2]
        else:
            # 如果没有完整实体，选择最长的实体
            target_entities = sorted(target_entities, key=lambda x: len(x[2]), reverse=True)
            target_span = target_entities[0]
            target_text = target_span[2]
    else:
        target_text = "NULL"

    # 选择最相关的Argument
    arg_entities = entities.get('ARG', [])
    if arg_entities:
        # 优先选择完整的短语
        complete_entities = [e for e in arg_entities if len(e[2]) > 2 and e[2].strip()]

        if complete_entities:
            # 选择最长的完整实体
            arg_entities = sorted(complete_entities, key=lambda x: len(x[2]), reverse=True)
            arg_span = arg_entities[0]
            arg_text = arg_span[2]
        else:
            # 如果没有完整实体，选择最长的实体
            arg_entities = sorted(arg_entities, key=lambda x: len(x[2]), reverse=True)
            arg_span = arg_entities[0]
            arg_text = arg_span[2]
    else:
        arg_text = "NULL"

    # 合并目标群体列表为字符串
    target_group_str = ', '.join(target_groups)

    # 只生成一个四元组
    quads.append(f"{target_text} | {arg_text} | {target_group_str} | {hate_label}")

    # 格式化输出
    output_str = " [SEP] ".join(quads) + " [END]"
    return output_str


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")
    train_data = load_data('train.json')
    test_data = load_data('test1.json')

    # 划分训练集和验证集 (80-20)
    np.random.seed(42)
    val_size = int(0.2 * len(train_data))
    indices = np.random.permutation(len(train_data))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_set = [train_data[i] for i in train_indices]
    val_set = [train_data[i] for i in val_indices]

    # 初始化快速分词器
    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)

    # 创建数据集
    print("创建数据集...")
    train_dataset = HateSpeechDataset(train_set, tokenizer, is_train=True)
    val_dataset = HateSpeechDataset(val_set, tokenizer, is_train=True)
    test_dataset = HateSpeechDataset(test_data, tokenizer, is_train=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # 初始化模型
    print("初始化模型...")
    model = HateSpeechModel(config)

    # 训练模型
    print("开始训练模型...")
    model = train_model(model, train_loader, val_loader, device, tokenizer)

    # 保存模型
    torch.save(model.state_dict(), 'hate_model3.pt')
    print("模型已保存到 hate_model3.pt")

    # 在测试集上进行预测
    print("在测试集上进行预测...")
    predictions = []
    model.to(device)

    for batch in tqdm(test_loader, desc="预测中"):
        texts = batch['text']
        ids = batch['id']

        for i, text in enumerate(texts):
            output = predict_quads(model, tokenizer, text, device)
            predictions.append(output)

    # 保存预测结果
    with open('demo3.txt', 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')

    print("预测结果已保存到 demo3.txt")


if __name__ == "__main__":
    main()