import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import pickle

class UserFriendlyAITrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=20)
        self.expected_outputs = {}
        self.model_dir = "saved_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.epochs = 10
        self.batch_size = 32
        self.lr = 0.001
        self.train_loader = None
        self.test_loader = None

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
                    raise ValueError(f"必须在{choices}中选择")
                
                return input_type(user_input)
            except ValueError as e:
                print(f"输入无效: {e}. 请重试")

    def configure(self, load_existing=False):
        """配置训练参数和对话模式"""
        if not load_existing:
            print("\n=== 训练参数配置 ===")
            self.epochs = self.get_input("训练轮数", int, self.epochs)
            self.batch_size = self.get_input("批大小", int, self.batch_size)
            self.lr = self.get_input("学习率", float, self.lr)

            print("\n=== 配置对话模式 ===")
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
        samples = []
        labels = []
        for idx, (pattern, _) in enumerate(self.expected_outputs.items()):
            samples.extend([f"{pattern} {i}" for i in range(10)])
            labels.extend([idx] * 10)
        
        X = self.vectorizer.fit_transform(samples).toarray()
        y = np.array(labels)
        
        # 奇怪的bug修复
        if X.shape[1] < 20:
            X = np.pad(X, ((0,0), (0, 20-X.shape[1])), 'constant')
        
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
            print("准备训练数据...")
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
            
        additional_epochs = self.get_input("请输入要追加的训练轮数", int, 5)
        self.train(additional_epochs=additional_epochs)

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
            
            # 加载模型后重新准备数据，这里太气人了。改了半天
            self._prepare_training_data()
            
            print(f"已从 {model_path} 加载模型")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

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

def main():
    trainer = UserFriendlyAITrainer()
    
    while True:
        print("\n=== 主菜单 ===")
        print("1. 训练新模型")
        print("2. 加载已有模型")
        print("3. 继续训练已有模型")
        print("4. 测试对话")
        print("5. 退出")
        print("6.千万别选择我!")
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
            trainer.predict()
        elif choice == '5':
            print("抱歉，请手动退出AuA")
        elif choice == '6':
            print("你不会以为真的有什么彩蛋吧?h h h!")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()