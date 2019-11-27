import tensorflow as tf
import model as ml
import data
import numpy as np
import os
import sys

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

from configs import DEFINES


DATA_OUT_PATH = './data_out/'

# Req. 1-5-1. bleu score 계산 함수
def bleu_compute(answer, preAnswer):
    return sentence_bleu(answer, preAnswer, smoothing_function=SmoothingFunction().method0)

# Req. 1-5-2. rouge score 계산 함수
def rouge_compute(answer,preAnswer):
    rouge = Rouge()
    return rouge.get_scores(answer, preAnswer)

# Req. 1-5-3. main 함수 구성
def main(self):
    data_out_path = os.path.join(os.getcwd(), DATA_OUT_PATH)
    os.makedirs(data_out_path, exist_ok=True)
    # 데이터를 통한 사전 구성 한다.
    char2idx, idx2char, vocabulary_length = data.load_voc()
    # 훈련 데이터와 테스트 데이터를 가져온다.
    train_q, train_a, test_q, test_a = data.load_data()

    # 훈련셋 인코딩 만드는 부분
    train_input_enc = data.enc_processing(train_q,char2idx)
    # 훈련셋 디코딩 입력 부분
    train_input_dec = data.dec_input_processing(train_a,char2idx)
    # 훈련셋 디코딩 출력 부분
    train_target_dec = data.dec_target_processing(train_a,char2idx)

    # 평가셋 인코딩 만드는 부분
    eval_input_enc = data.enc_processing(test_q,char2idx)
    # 평가셋 인코딩 만드는 부분
    eval_input_dec = data.dec_input_processing(test_a,char2idx)
    # 평가셋 인코딩 만드는 부분
    eval_target_dec = data.dec_target_processing(test_a,char2idx)

    # 현재 경로'./'에 현재 경로 하부에
    # 체크 포인트를 저장한 디렉토리를 설정한다.
    check_point_path = os.path.join(os.getcwd(), DEFINES.check_point_path)
    # 디렉토리를 만드는 함수이며 두번째 인자 exist_ok가
    # True이면 디렉토리가 이미 존재해도 OSError가
    # 발생하지 않는다.
    # exist_ok가 False이면 이미 존재하면
    # OSError가 발생한다.
    os.makedirs(check_point_path, exist_ok=True)

    # 에스티메이터 구성한다.
    classifier = tf.estimator.Estimator(
        model_fn=ml.Model,  # 모델 등록한다.
        model_dir=DEFINES.check_point_path,  # 체크포인트 위치 등록한다.
        params={  # 모델 쪽으로 파라메터 전달한다.
            'hidden_size': DEFINES.hidden_size, # 가중치 크기 설정한다.
            'learning_rate': DEFINES.learning_rate, # 학습율 설정한다. 
            'vocabulary_length': vocabulary_length, # 딕셔너리 크기를 설정한다.
            'embedding_size': DEFINES.embedding_size, # 임베딩 크기를 설정한다.
            'max_sequence_length': DEFINES.max_sequence_length
        })


    # 학습 실행
    classifier.train(input_fn=lambda: data.train_input_fn(
        train_input_enc, train_input_dec, train_target_dec, DEFINES.batch_size), steps=DEFINES.train_steps)

    eval_result = classifier.evaluate(input_fn=lambda: data.eval_input_fn(
        eval_input_enc, eval_input_dec, eval_target_dec, DEFINES.batch_size))

    print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # 테스트용 데이터 만드는 부분이다.
    # 인코딩 부분 만든다. 테스트용으로 ["가끔 궁금해"] 값을 넣어 형성된 대답과 비교를 한다.
    predic_input_enc = data.enc_processing(["가끔 궁금해"], char2idx)
    # 학습 과정이 아니므로 디코딩 입력은
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    predic_input_dec = data.dec_input_processing([""], char2idx)
    # 학습 과정이 아니므로 디코딩 출력 부분도
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    predic_target_dec = data.dec_target_processing([""], char2idx)

    predictions = classifier.predict(
        input_fn=lambda: data.eval_input_fn(predic_input_enc, predic_input_dec, predic_target_dec, 1))

    answer = data.pred_next_string(predictions, idx2char)

    # 예측한 값을 인지 할 수 있도록
    # 텍스트로 변경하는 부분이다.
    print("answer: ", answer)
    print("Bleu score: ", bleu_compute("그 사람도 그럴 거예요.", answer))
    print("Rouge score: ", rouge_compute("그 사람도 그럴 거예요.", answer))


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main)

tf.compat.v1.logging.set_verbosity
