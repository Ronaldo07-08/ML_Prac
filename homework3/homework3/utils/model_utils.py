def create_layer_config(num_hidden_layers, hidden_size=128, use_reg=False):
    """
    Генерирует конфигурацию слоев для FullyConnectedModel.
    """
    layers = []
    for _ in range(num_hidden_layers):
        layers.append({"type": "linear", "size": hidden_size})
        if use_reg:
            layers.append({"type": "batch_norm"})
            layers.append({"type": "relu"})
            layers.append({"type": "dropout", "rate": 0.3})
        else:
            layers.append({"type": "relu"})
    return layers

def create_custom_width_config(layer_sizes, use_reg=False):
    """
    Генерирует конфиг слоев с явно заданными размерами (шириной) каждого скрытого слоя.
    Принимает список, например: [64, 32, 16]
    """
    layers = []
    for size in layer_sizes:
        layers.append({"type": "linear", "size": size})
        if use_reg:
            layers.append({"type": "batch_norm"})
            layers.append({"type": "relu"})
            layers.append({"type": "dropout", "rate": 0.3})
        else:
            layers.append({"type": "relu"})
    return layers

def build_reg_config(sizes=[256, 128, 64], reg_type='none', dropout_rate=0.5):
    """
    Генерирует конфиг с нужным типом регуляризации.
    reg_type может быть: 'none', 'dropout', 'batchnorm', 'both'
    """
    layers = []
    for size in sizes:
        layers.append({"type": "linear", "size": size})
        
        if reg_type in ['batchnorm', 'both']:
            layers.append({"type": "batch_norm"})
            
        layers.append({"type": "relu"})

        if reg_type in ['dropout', 'both']:
            layers.append({"type": "dropout", "rate": dropout_rate})
            
    return layers