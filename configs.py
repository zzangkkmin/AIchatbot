import tensorflow as tf

tf.app.flags.DEFINE_string('f', '', 'kernel') #
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')  # 배치 크기
tf.app.flags.DEFINE_integer('train_steps', 20000, 'train steps')  # 학습 에포크
tf.app.flags.DEFINE_float('dropout_width', 0.5, 'dropout width')  # 드롭아웃 크기
tf.app.flags.DEFINE_integer('embedding_size', 128, 'embedding size')  # 임베딩 크기
tf.app.flags.DEFINE_integer('hidden_size', 128, 'weights size') # 가중치 크기
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')  # 학습률
tf.app.flags.DEFINE_integer('shuffle_seek', 1000, 'shuffle random seek')  # 셔플 시드값
tf.app.flags.DEFINE_integer('max_sequence_length', 25, 'max sequence length')  # 시퀀스 길이
tf.app.flags.DEFINE_integer('ff_input', 2, 'ff input size')  # ff label 크기
tf.app.flags.DEFINE_integer('attention_head_size', 4, 'attn head size')  # 멀티 헤드 크기
tf.app.flags.DEFINE_integer('layers_size', 2, 'layer size')  # 학습 속도 및 성능 튜닝
tf.app.flags.DEFINE_string('data_path', './data_in/ChatBotData.csv', 'data path')  # 데이터 위치
tf.app.flags.DEFINE_string('vocabulary_path', './data_out/vocabularyData.voc', 'vocabulary path')  # 사전 위치
tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point', 'check point path')  # 체크 포인트 위치
tf.app.flags.DEFINE_boolean('tokenize_as_morph', False, 'set morph tokenize') # 형태소에 따른 토크나이징 사용 유무
tf.app.flags.DEFINE_boolean('embedding', True, 'Use Embedding flag') # 임베딩 유무 설정
tf.app.flags.DEFINE_boolean('multilayer', True, 'Use Multi RNN Cell') # 멀티 RNN 유무
tf.app.flags.DEFINE_integer('query_dimention', 128, 'query dimention')
tf.app.flags.DEFINE_integer('key_dimention', 128, 'key dimention')
tf.app.flags.DEFINE_integer('heads_size', 4, 'heads size')
tf.app.flags.DEFINE_boolean('conv_1d_layer', True, 'set conv 1d layer')
tf.app.flags.DEFINE_boolean('xavier_embedding', True, 'set init xavier embedding')
# tf.app.flags.DEFINE_boolean('xavier_initializer', True, 'set xavier initializer')  # 형태소에 따른 토크나이징 사용 유무


# Define FLAGS
DEFINES = tf.app.flags.FLAGS