from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Reshape, Input, BatchNormalization
from tensorflow.keras.layers import Activation, add, multiply, Dropout

def swish(x):
    return Activation('swish')(x)

def mbconv_block(input_x, filters, stride, expansion_factor, se_ratio=4):
    """
    Khối MBConv chuẩn: Expansion -> Depthwise -> Squeeze-and-Excitation -> Projection -> Skip Connection
    """
    in_channels = input_x.shape[-1]
    x = input_x
    
    # 1. Expansion phase (Mở rộng số kênh)
    if expansion_factor != 1:
        x = Conv2D(in_channels * expansion_factor, kernel_size=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = swish(x)
        
    # 2. Depthwise phase (Trích xuất đặc trưng không gian)
    x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = swish(x)
    
    # 3. Squeeze and Excitation phase (Cơ chế chú ý)
    if se_ratio > 0:
        expanded_channels = in_channels * expansion_factor
        se = GlobalAveragePooling2D()(x)
        se = Reshape((1, 1, expanded_channels))(se)
        # Nén kênh (Squeeze)
        se = Dense(max(1, expanded_channels // se_ratio), activation='swish')(se)
        # Kích thích (Excitation)
        se = Dense(expanded_channels, activation='sigmoid')(se)
        x = multiply([x, se])
        
    # 4. Projection phase (Chiếu lại về số kênh yêu cầu)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    # LƯU Ý: Không dùng activation function ở bước projection
    
    # 5. Skip connection (Chỉ thực hiện khi đầu vào và đầu ra khớp kích thước)
    if stride == 1 and in_channels == filters:
        x = add([input_x, x])
        
    return x

def createLayers(input_x, out_classes):
    # Phần Đầu (Head): Giữ nguyên kích thước 32x32 lâu hơn để trích xuất dồi dào đặc trưng cơ bản
    x = Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False)(input_x)
    x = BatchNormalization()(x)
    x = swish(x)

    # Phần Thân (Body): Các khối MBConv
    # Định dạng cấu hình: [filters, stride, expansion_factor, số lần lặp lại block]
    block_configs = [
        [16,  1, 1, 1], # Cố định số kênh ban đầu
        [32,  2, 4, 2], # Giảm size xuống 16x16, tăng channel
        [64,  2, 4, 3], # Giảm size xuống 8x8, tăng channel sâu hơn
        [128, 2, 6, 3], # Giảm size xuống 4x4, học đặc trưng phức tạp
        [256, 2, 6, 1]  # Giảm size xuống 2x2, đỉnh của tháp trích xuất
    ]

    for filters, stride, expansion, repeats in block_configs:
        for i in range(repeats):
            # Chỉ layer đầu tiên trong nhóm mới áp dụng stride (để giảm kích thước)
            current_stride = stride if i == 0 else 1
            x = mbconv_block(x, filters=filters, stride=current_stride, expansion_factor=expansion)

    # Phần Đuôi (Tail): Bộ phân loại
    x = Conv2D(512, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = swish(x)
    
    x = GlobalAveragePooling2D()(x)
    # Thêm Dropout để tránh Overfitting (rất quan trọng khi tăng dung lượng mạng)
    x = Dropout(0.3)(x) 
    x = Dense(out_classes, activation='softmax')(x)

    return x

if __name__ == '__main__':
    inputs = Input(shape=(32, 32, 1))
    outputs = createLayers(inputs, 94)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()