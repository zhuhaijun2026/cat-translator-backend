#!/bin/bash
# 猫语翻译器 - 本地开发环境搭建脚本

echo "🐱 猫语翻译器 - 环境搭建"
echo "=========================="

# 1. 检查 Python
echo ""
echo "📋 Step 1: 检查 Python 环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ 找到 $PYTHON_VERSION"
else
    echo "❌ 未找到 Python3，请先安装 Python 3.10+"
    echo "   下载地址：https://www.python.org/downloads/"
    exit 1
fi

# 2. 创建虚拟环境
echo ""
echo "📋 Step 2: 创建 Python 虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境已创建"
else
    echo "✅ 虚拟环境已存在"
fi

# 3. 激活虚拟环境并安装依赖
echo ""
echo "📋 Step 3: 安装 Python 依赖..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ 依赖安装完成"

# 4. 创建数据目录
echo ""
echo "📋 Step 4: 创建数据目录..."
mkdir -p data/raw/CatMeows
mkdir -p data/augmented
mkdir -p data/models
mkdir -p static/audio/cats
echo "✅ 目录创建完成"

# 5. 检查 CatMeows 数据集
echo ""
echo "📋 Step 5: 检查猫叫数据集..."
if [ -z "$(ls -A data/raw/CatMeows 2>/dev/null)" ]; then
    echo "⚠️  CatMeows 数据集未下载"
    echo "   请手动下载："
    echo "   1. 访问 https://zenodo.org/record/4008297"
    echo "   2. 下载 dataset.zip"
    echo "   3. 解压到 data/raw/CatMeows/"
    echo ""
    echo "   或者使用 HuggingFace："
    echo "   pip install datasets"
    echo "   python -c \"from datasets import load_dataset; ds = load_dataset('zeddez/CatMeows')\""
else
    echo "✅ CatMeows 数据集已存在"
fi

echo ""
echo "🐱 环境搭建完成！"
echo ""
echo "下一步操作："
echo "  1. 启动后端：source venv/bin/activate && python main.py"
echo "  2. 测试 API：curl http://localhost:8000/docs"
echo "  3. 在 HBuilderX 中打开 cat-translator 前端项目"
echo ""
