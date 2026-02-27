# Создайте файл check_cuda.py
import torch
import sys
import subprocess
import os

print("=" * 60)
print("ПРОВЕРКА CUDA И ПОДДЕРЖКИ GPU")
print("=" * 60)

# 1. Базовая проверка PyTorch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступно в PyTorch: {torch.cuda.is_available()}")
print(f"CUDA версия в PyTorch: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

# 2. Проверка установки CUDA в системе
print(f"\nПроверка системы:")
try:
    # Пробуем найти nvidia-smi
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ NVIDIA SMI найден")
        # Выводим информацию о GPU
        lines = result.stdout.split('\n')
        for line in lines[:10]:
            if line.strip():
                print(f"  {line}")
    else:
        print("❌ NVIDIA SMI не найден или не запускается")
except Exception as e:
    print(f"❌ Ошибка при запуске nvidia-smi: {e}")

# 3. Проверка CUDA через PyTorch детально
if torch.cuda.is_available():
    print(f"\n✅ CUDA ДОСТУПНО!")
    print(f"  Количество GPU: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}:")
        print(f"    Имя: {torch.cuda.get_device_name(i)}")
        print(f"    Память: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"    Вычислительная способность: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Проверка текущего устройства
    print(f"\n  Текущее устройство: {torch.cuda.current_device()}")
    print(f"  Текущее имя устройства: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print(f"\n❌ CUDA НЕ ДОСТУПНО")
    
    # Проверяем переменные окружения
    print(f"\n  Проверка переменных окружения:")
    cuda_paths = [
        "CUDA_PATH",
        "CUDA_HOME",
        "PATH"
    ]
    
    for var in cuda_paths:
        value = os.environ.get(var, "Не установлена")
        if var == "PATH":
            # Проверяем наличие CUDA в PATH
            path_parts = value.split(';')
            cuda_in_path = any('cuda' in part.lower() for part in path_parts)
            print(f"    {var}: {'✅ Содержит CUDA' if cuda_in_path else '❌ Не содержит CUDA'}")
        else:
            exists = os.path.exists(value) if os.path.isabs(value) else False
            print(f"    {var}: {value} {'✅ Существует' if exists else '❌ Не существует'}")

# 4. Проверка установленного PyTorch
print(f"\nПроверка установки PyTorch:")
try:
    import torch.utils.collect_env
    print("  Информация об окружении:")
    env_info = torch.utils.collect_env.get_pretty_env_info()
    for line in env_info.split('\n'):
        if 'CUDA' in line or 'GPU' in line or 'cuda' in line.lower():
            print(f"    {line}")
except:
    print("  Не удалось получить детальную информацию")

print("\n" + "=" * 60)