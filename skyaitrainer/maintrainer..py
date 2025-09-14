import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import pickle
import sqlite3
import csv
import json
import requests
from bs4 import BeautifulSoup
import re

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleClassifier(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.view(1, -1)
        return self.net(x)

class UserFriendlyAITrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=20)
        self.expected_outputs = {}
        self.model_dir = "saved_models"
        self.data_dir = "training_data"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        self.epochs = 10
        self.batch_size = 32
        self.lr = 0.001
        self.train_loader = None
        self.test_loader = None
        self.criterion = nn.CrossEntropyLoss()

    def get_input(self, prompt, input_type=str, default=None, choices=None):
        while True:
            try:
                if choices:
                    prompt += f" ({'/'.join(choices)})"
                if default is not None:
                    prompt += f" [{default}]"
                prompt += ": "
                
                user_input = input(prompt).strip()
                if not user_input and default is not None:
                    return default
                
                if choices and user_input.lower() not in [c.lower() for c in choices]:
                    raise ValueError(f"您必须在{choices}中选择")
                
                return input_type(user_input)
            except ValueError as e:
                print(f"错误：输入无效: {e}. 请重试")

    def load_from_database(self):
        """从数据库或文件加载训练数据"""
        print("\n=== 数据加载选项 ===")
        print("1. SQLite数据库 (.db)")
        print("2. CSV文件")
        print("3. JSON文件")
        print("4. 手动输入")
        
        choice = self.get_input("请选择数据来源", choices=['1', '2', '3', '4'], default='4')
        
        if choice == '1':
            return self._load_from_sqlite()
        elif choice == '2':
            return self._load_from_csv()
        elif choice == '3':
            return self._load_from_json()
        else:
            return None

    def _load_from_sqlite(self):
        """从SQLite数据库加载数据"""
        db_files = [f for f in os.listdir(self.data_dir) if f.endswith('.db')]
        if not db_files:
            print(f"在 {self.data_dir} 目录下未找到.db文件")
            return None
            
        print("\n现在可用的数据库文件:")
        for i, f in enumerate(db_files, 1):
            print(f"{i}. {f}")
        
        file_idx = self.get_input("选择数据库文件", int) - 1
        if file_idx < 0 or file_idx >= len(db_files):
            print("这是一个无效选择")
            return None
            
        db_path = os.path.join(self.data_dir, db_files[file_idx])
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 获取所有表
            tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            print("\n现在可用的表:")
            for i, (table,) in enumerate(tables, 1):
                print(f"{i}. {table}")
            
            table_idx = self.get_input("选择表", int) - 1
            if table_idx < 0 or table_idx >= len(tables):
                print("这是一个无效选择")
                return None
                
            table_name = tables[table_idx][0]
            
            # 获取列名
            columns = [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name});").fetchall()]
            print("\n现在可用的列:")
            for i, col in enumerate(columns, 1):
                print(f"{i}. {col}")
            
            print("\n请选择包含触发消息和回复的列:")
            pattern_col = columns[self.get_input("触发消息列", int) - 1]
            response_col = columns[self.get_input("回复内容列", int) - 1]
            
            # 查询数据
            data = cursor.execute(f"SELECT {pattern_col}, {response_col} FROM {table_name};").fetchall()
            
            if not data:
                print("此表中没有数据")
                return None
                
            # 转换为字典
            patterns = {}
            for pattern, response in data:
                if pattern and response:
                    patterns[pattern] = response
                    
            return patterns if patterns else None
            
        except Exception as e:
            print(f"来自数据库错误: {e}")
            return None
        finally:
            conn.close()

    def _load_from_csv(self):
        """从CSV文件加载数据"""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"在 {self.data_dir} 目录下未找到.csv文件")
            return None
            
        print("\n可用的CSV文件:")
        for i, f in enumerate(csv_files, 1):
            print(f"{i}. {f}")
        
        file_idx = self.get_input("选择CSV文件", int) - 1
        if file_idx < 0 or file_idx >= len(csv_files):
            print("无效选择")
            return None
            
        csv_path = os.path.join(self.data_dir, csv_files[file_idx])
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                print("\n现在可用的列:")
                for i, header in enumerate(headers, 1):
                    print(f"{i}. {header}")
                
                print("\n请选择包含触发消息和回复的列:")
                pattern_col = self.get_input("触发消息列", int) - 1
                response_col = self.get_input("回复内容列", int) - 1
                
                patterns = {}
                for row in reader:
                    if len(row) > max(pattern_col, response_col):
                        pattern = row[pattern_col].strip()
                        response = row[response_col].strip()
                        if pattern and response:
                            patterns[pattern] = response
                
                return patterns if patterns else None
                
        except Exception as e:
            print(f"CSV文件错误: {e}")
            return None

    def _load_from_json(self):
        """从JSON文件加载数据"""
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        if not json_files:
            print(f"在 {self.data_dir} 目录下未找到.json文件")
            return None
            
        print("\n可用的JSON文件:")
        for i, f in enumerate(json_files, 1):
            print(f"{i}. {f}")
        
        file_idx = self.get_input("选择JSON文件", int) - 1
        if file_idx < 0 or file_idx >= len(json_files):
            print("这是一个无效选择")
            return None
            
        json_path = os.path.join(self.data_dir, json_files[file_idx])
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, dict):
                    # 直接使用字典格式
                    return data
                elif isinstance(data, list):
                    # 处理列表格式
                    patterns = {}
                    for item in data:
                        if isinstance(item, dict) and 'pattern' in item and 'response' in item:
                            patterns[item['pattern']] = item['response']
                    return patterns if patterns else None
                else:
                    print("不支持的JSON格式")
                    return None
                    
        except Exception as e:
            print(f"JSON文件错误: {e}")
            return None

    def configure(self, load_existing=False):
        """配置训练参数和对话模式"""
        if not load_existing:
            print("\n=== 训练参数配置 ===")
            self.epochs = self.get_input("训练轮数", int, self.epochs)
            self.batch_size = self.get_input("批大小", int, self.batch_size)
            self.lr = self.get_input("学习率", float, self.lr)

            print("\n=== 配置对话模式 ===")
            print("1. 从文件/数据库导入")
            print("2. 手动输入")
            
            choice = self.get_input("选择配置方式", choices=['1', '2'], default='2')
            
            if choice == '1':
                patterns = self.load_from_database()
                if patterns:
                    self.expected_outputs.update(patterns)
                    print(f"成功导入 {len(patterns)} 条对话模式")
            else:
                while True:
                    pattern = self.get_input("输入触发消息(如'天气'，输入q完成)", default="q")
                    if pattern.lower() == 'q':
                        if not self.expected_outputs:
                            print("至少需要配置一个回复模式!")
                            continue
                        break
                    response = self.get_input("期望回复内容")
                    self.expected_outputs[pattern] = response

        # 生成训练数据
        self._prepare_training_data()

        # 初始化模型
        if not load_existing:
            self.model = SimpleClassifier(
                input_size=20,
                hidden_size=64,
                output_size=len(self.expected_outputs)
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.criterion = nn.CrossEntropyLoss()

    def _prepare_training_data(self):
        """准备训练数据"""
        if not self.expected_outputs:
            print("没有可用的训练数据")
            return
            
        samples = []
        labels = []
        for idx, (pattern, _) in enumerate(self.expected_outputs.items()):
            samples.extend([f"{pattern} {i}" for i in range(10)])
            labels.extend([idx] * 10)
        
        X = self.vectorizer.fit_transform(samples).toarray()
        y = np.array(labels)
        
        if X.shape[1] < 20:
            X = np.pad(X, ((0,0), (0, 20-X.shape[1])), 'constant')
        elif X.shape[1] > 20:
            X = X[:, :20]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.train_loader = DataLoader(
            TextDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            TextDataset(X_test, y_test),
            batch_size=self.batch_size
        )

    def train(self, additional_epochs=None):
        """训练模型"""
        if self.train_loader is None:
            self._prepare_training_data()
            
        total_epochs = additional_epochs if additional_epochs else self.epochs
        print(f"\n=== 开始训练 ({total_epochs}轮) ===")
        
        for epoch in range(total_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, labels in self.train_loader:
                inputs = inputs.view(-1, 20).to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            val_acc = self.evaluate()
            print(f"Epoch {epoch+1}/{total_epochs}, Loss: {total_loss/len(self.train_loader):.4f}, "
                  f"Train Acc: {acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        self.save_model("final")
        print("训练完成!")

    def continue_training(self):
        """继续训练已有模型"""
        if not hasattr(self, 'model') or self.model is None:
            print("请先加载模型")
            return
            
        print("\n=== 继续训练配置 ===")
        print("当前配置:")
        print(f"- 训练轮数: {self.epochs}")
        print(f"- 批大小: {self.batch_size}")
        print(f"- 学习率: {self.lr}")
        print(f"- 已配置模式: {len(self.expected_outputs)}个")
        
        # 训练参数配置
        print("\n配置训练参数(直接回车保持当前值):")
        self.epochs = self.get_input("训练轮数", int, self.epochs)
        self.batch_size = self.get_input("此批大小", int, self.batch_size)
        self.lr = self.get_input("AI学习率", float, self.lr)
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        
        # 对话模式配置
        print("\n当前对话模式:")
        for i, (pattern, response) in enumerate(self.expected_outputs.items(), 1):
            print(f"{i}. {pattern} -> {response}")
            
        modify_patterns = self.get_input("\n是否修改对话模式? (y/n)", choices=['y', 'n'], default='n')
        if modify_patterns.lower() == 'y':
            print("\n=== 修改对话模式 ===")
            print("1. 从文件/数据库导入")
            print("2. 手动修改")
            print("3. 保持当前模式")
            
            choice = self.get_input("请选择操作", choices=['1', '2', '3'], default='3')
            
            if choice == '1':
                patterns = self.load_from_database()
                if patterns:
                    self.expected_outputs.update(patterns)
                    print(f"成功导入 {len(patterns)} 条对话模式")
            elif choice == '2':
                print("\n1. 添加新模式")
                print("2. 删除现有模式")
                print("3. 修改现有回复")
                print("4. 取消")
                
                sub_choice = self.get_input("请选择操作", choices=['1', '2', '3', '4'], default='4')
                
                if sub_choice == '1':
                    while True:
                        pattern = self.get_input("输入新触发消息(如'天气'，输入q完成)", default="q")
                        if pattern.lower() == 'q':
                            break
                        response = self.get_input("期望回复内容")
                        self.expected_outputs[pattern] = response
                elif sub_choice == '2':
                    patterns = list(self.expected_outputs.keys())
                    for i, p in enumerate(patterns, 1):
                        print(f"{i}. {p}")
                    to_delete = self.get_input("输入要删除的模式编号(输入q取消)", default="q")
                    if to_delete.lower() != 'q':
                        try:
                            idx = int(to_delete) - 1
                            if 0 <= idx < len(patterns):
                                del self.expected_outputs[patterns[idx]]
                                print("模式已删除")
                        except ValueError:
                            print("无效输入")
                elif sub_choice == '3':
                    patterns = list(self.expected_outputs.keys())
                    for i, p in enumerate(patterns, 1):
                        print(f"{i}. {p} -> {self.expected_outputs[p]}")
                    to_modify = self.get_input("输入要修改的模式编号(输入q取消)", default="q")
                    if to_modify.lower() != 'q':
                        try:
                            idx = int(to_modify) - 1
                            if 0 <= idx < len(patterns):
                                new_response = self.get_input("输入新的回复内容", 
                                                             default=self.expected_outputs[patterns[idx]])
                                self.expected_outputs[patterns[idx]] = new_response
                                print("回复已更新")
                        except ValueError:
                            print("无效输入")
        
        # 重新准备数据
        self._prepare_training_data()
        
        # 输出层大小变化，需要调整模型
        if len(self.expected_outputs) != self.model.net[-1].out_features:
            print("检测到输出层大小变化，调整模型结构...")
            self.model = SimpleClassifier(
                input_size=20,
                hidden_size=64,
                output_size=len(self.expected_outputs)
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 开始训练
        self.train()

    def evaluate(self):
        """评估模型"""
        if self.test_loader is None:
            self._prepare_training_data()
            
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.view(-1, 20).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

    def predict(self):
        """对话测试"""
        if not hasattr(self, 'model') or self.model is None:
            print("请先训练或加载模型")
            return
            
        print("\n=== 对话测试(输入q退出) ===")
        while True:
            message = input("\n输入消息: ").strip()
            if message.lower() == 'q':
                break
            
            try:
                features = self.vectorizer.transform([message]).toarray()
                if features.shape[1] < 20:
                    features = np.pad(features, ((0,0), (0, 20-features.shape[1])), 'constant')
                elif features.shape[1] > 20:
                    features = features[:, :20]
                
                input_tensor = torch.FloatTensor(features).view(1, 20).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    predicted = torch.argmax(output).item()
                
                if predicted < len(self.expected_outputs):
                    pattern = list(self.expected_outputs.keys())[predicted]
                    print(f"识别的消息模式: {pattern}")
                    print(f"回复: {self.expected_outputs[pattern]}")
                else:
                    print("未识别此消息模式")
            except Exception as e:
                print(f"处理错误: {e}")

    def save_model(self, name):
        """保存模型"""
        model_path = os.path.join(self.model_dir, f"{name}_model.pth")
        config_path = os.path.join(self.model_dir, f"{name}_config.pkl")
        
        torch.save(self.model.state_dict(), model_path)
        
        config = {
            'expected_outputs': self.expected_outputs,
            'vectorizer': self.vectorizer,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr
        }
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"模型已保存到 {model_path} 和 {config_path}")

    def load_model(self, name="final"):
        """加载模型"""
        model_path = os.path.join(self.model_dir, f"{name}_model.pth")
        config_path = os.path.join(self.model_dir, f"{name}_config.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"找不到{name}模型或配置文件")
            return False
        
        try:
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                self.expected_outputs = config['expected_outputs']
                self.vectorizer = config['vectorizer']
                self.epochs = config.get('epochs', 10)
                self.batch_size = config.get('batch_size', 32)
                self.lr = config.get('lr', 0.001)
            
            self.model = SimpleClassifier(
                input_size=20,
                hidden_size=64,
                output_size=len(self.expected_outputs)
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(model_path))
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.criterion = nn.CrossEntropyLoss()
            
            # 加载模型后重新准备数据
            self._prepare_training_data()
            
            print(f"已从 {model_path} 中加载模型")
            return True
        except Exception as e:
            print(f"错误：加载模型失败: {e}")
            return False

    def web_search_train(self):
        """联网搜索并训练模型"""
        print("\n=== 联网搜索训练 ===")
        while True:
            search_query = input("\n请输入要搜索并训练的内容(输入q退出): ").strip()
            if search_query.lower() == 'q':
                break
            
            try:
                print(f"正在搜索: {search_query}...")
                search_results = self._fetch_search_results(search_query)
                
                if not search_results:
                    print("未找到相关搜索结果")
                    continue
                
                print("提取搜索结果中的文本内容...")
                texts = self._extract_text_from_results(search_results)
                
                if not texts:
                    print("未能从搜索结果中提取有效文本")
                    continue
                
                print(f"找到 {len(texts)} 条相关文本，添加到训练数据...")
                for i, text in enumerate(texts[:10]):
                    pattern = f"{search_query}_{i+1}"
                    self.expected_outputs[pattern] = text[:200]
                
                self._prepare_training_data()
                
                if self.model is None:
                    self._initialize_model()
                elif len(self.expected_outputs) != self.model.net[-1].out_features:
                    print("检测到输出层大小变化，调整模型结构...")
                    self._initialize_model()
                
                print("开始训练...")
                self.train(additional_epochs=5)
                
                print("\n训练完成!")
                continue_train = input("还需要继续联网训练吗? (按q退出，回车键继续): ").strip()
                if continue_train.lower() == 'q':
                    break
                    
            except Exception as e:
                print(f"联网训练出错: {e}")
                continue

    def _fetch_search_results(self, query):
        """获取搜索结果"""
        try:
            url = f"https://cn.bing.com/search?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"搜索出错: {e}")
            return None

    def _extract_text_from_results(self, html):
        """从搜索结果中提取文本"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for script in soup(["script", "style", "iframe", "noscript"]):
                script.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            
            paragraphs = [p for p in text.split('. ') if len(p) > 20]
            return paragraphs
        except Exception as e:
            print(f"提取文本出错: {e}")
            return None

    def _initialize_model(self):
        """初始化模型"""
        self.model = SimpleClassifier(
            input_size=20,
            hidden_size=64,
            output_size=len(self.expected_outputs)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

def main():
    trainer = UserFriendlyAITrainer()
    
    while True:
        print("\n=== 主菜单 ===")
        print("1. 训练新模型")
        print("2. 加载已有模型")
        print("3. 继续训练已有模型")
        print("4. 联网搜索训练")
        print("5. 测试对话")
        print("6. 退出")
        print("7.千万别选择我!")
        print("本程序简陋至极，还望包涵\n by:kj23")
        
        choice = input("请选择: ").strip()
        
        if choice == '1':
            trainer.configure()
            trainer.train()
        elif choice == '2':
            if trainer.load_model():
                print("模型加载成功")
        elif choice == '3':
            trainer.continue_training()
        elif choice == '4':
            trainer.web_search_train()
        elif choice == '5':
            trainer.predict()
        elif choice == '6':
            print("请输入Ctrl+C手动退出")
        elif choice == '7':
            print("这里...没有...彩蛋...哦！")
            break
        else:
            print("错误：无效选择，请重新输入")

if __name__ == "__main__":
    main()