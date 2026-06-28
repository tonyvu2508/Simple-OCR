import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, 
    Dense, Reshape, BatchNormalization, Activation, 
    add, multiply, Dropout
)

def swish(x):
    """Hàm kích hoạt Swish: mượt hơn ReLU, giúp mô hình sâu hội tụ tốt hơn."""
    return Activation('swish')(x)

def mbconv_block(input_x, filters, stride, expansion_factor, se_ratio=4):
    """
    Khối Mobile Inverted Bottleneck Conv (MBConv) tích hợp Squeeze-and-Excitation.
    """
    in_channels = input_x.shape[-1]
    x = input_x
    
    # 1. Expansion phase (Mở rộng số kênh đặc trưng)
    if expansion_factor != 1:
        x = Conv2D(in_channels * expansion_factor, kernel_size=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = swish(x)
        
    # 2. Depthwise phase (Trích xuất đặc trưng không gian)
    x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = swish(x)
    
    # 3. Squeeze and Excitation phase (Cơ chế chú ý - Attention)
    if se_ratio > 0:
        expanded_channels = in_channels * expansion_factor
        se = GlobalAveragePooling2D()(x)
        se = Reshape((1, 1, expanded_channels))(se)
        # Squeeze (Nén)
        se = Dense(max(1, expanded_channels // se_ratio), activation='swish')(se)
        # Excitation (Kích thích)
        se = Dense(expanded_channels, activation='sigmoid')(se)
        x = multiply([x, se])
        
    # 4. Projection phase (Chiếu lại về số kênh yêu cầu)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # 5. Skip connection (Cộng tắt - chỉ hoạt động khi cùng kích thước)
    if stride == 1 and in_channels == filters:
        x = add([input_x, x])
        
    return x

def MinimNet(input_shape=(32, 32, 1), num_classes=94):
    """
    Hàm khởi tạo kiến trúc mạng MinimNet phiên bản tối đa hóa độ chính xác.
    """
    inputs = Input(shape=input_shape)
    
    # --- PHẦN 1: HEAD (Khởi đầu) ---
    x = Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = swish(x)

    # --- PHẦN 2: BODY (Thân mạng) ---
    # Cấu hình: [số_kênh_đầu_ra, stride, hệ_số_mở_rộng, số_lần_lặp]
    block_configs = [
        [16,  1, 1, 1], # Cố định kích thước 32x32
        [32,  2, 4, 2], # Giảm xuống 16x16
        [64,  2, 4, 3], # Giảm xuống 8x8
        [128, 2, 6, 3], # Giảm xuống 4x4
        [256, 1, 6, 1]  # Giữ nguyên kích thước 4x4 (thay vì giảm xuống 2x2)
    ]

    for filters, stride, expansion, repeats in block_configs:
        for i in range(repeats):
            # Chỉ layer đầu tiên trong nhóm block mới áp dụng stride để giảm size
            current_stride = stride if i == 0 else 1
            x = mbconv_block(x, filters=filters, stride=current_stride, expansion_factor=expansion)

    # --- PHẦN 3: TAIL (Phần đuôi phân loại) ---
    x = Conv2D(512, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = swish(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x) # Tắt ngẫu nhiên 30% nơ-ron để chống Overfitting
    outputs = Dense(num_classes, activation='softmax')(x)

    # Đóng gói thành mô hình Keras
    model = Model(inputs=inputs, outputs=outputs, name="MinimNet_MaxAcc")
    return model