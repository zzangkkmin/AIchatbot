import data
import model as ml
import tensorflow as tf
import sqlite3

import json
import os

import pickle
import numpy as np
import db_util as db
import predict as pd

from flask import g
from threading import Thread
from configs import DEFINES
from flask import Flask, request, make_response, Response
from slack import WebClient
from slackeventsapi import SlackEventAdapter


# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-724021847745-731702713797-axJrESBdAVNfzCQlNhwjevg8"
SLACK_SIGNING_SECRET = "ee88404028007abaffb6b978f09a4074"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Req. 2-2-1 대답 예측 함수 구현
def predict(question):
    answer = pd.predict(question)
    return answer
    
# Req 2-2-2. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장
    

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    
    if request.headers.get('X-slack-Retry-Num') != None:
        return make_response("Ignore retries",200)

    user = event_data["event"]["user"]
    channel = event_data["event"]["channel"]
    question = event_data["event"]["text"]

    # predict 호출
    question = question.replace('<@UMHLNLZPF>', '').strip()
    answer = predict(question)
    
    # DB에 데이터 삽입
    db.insert(question, answer, user)
    
    message_attachments = [
            {
                "text": answer,
                "callback_id": "answer",
                "color": "#EBB424",
                "attachment_type": "default",
                "actions": [
                    {
                        "name": "show_conv",
                        "text": "대화내역 보기",
                        "type": "button"
                    },
                    {
                        "name": "delete_conv",
                        "text": "대화내역 삭제",
                        "type": "button"
                    }
                ]
            }
        ]
    
    slack_web_client.chat_postMessage(
        channel=channel,
        attachments=json.dumps(message_attachments)
    )

@app.route("/after_button", methods=["POST"])
def respond():
    slack_payload = json.loads(request.form.get("payload"))

    action = slack_payload["actions"][0]["name"]

    bot = slack_payload["original_message"]["username"]
    username = slack_payload["user"]["name"]
    userid = slack_payload["user"]["id"]
    answer = ""

    if (action == "show_conv"):
        conversation = db.findByUserName(userid)

        if(len(conversation) == 0):
            answer = "대화내역이 존재하지 않습니다."
        else:
            for row in conversation:
                time = row[4].split(" ")
                answer += "[" + time[0] + "][" + username + "] : " + row[1] + "\n"
                answer += "[" + time[0] + "][" + bot + "] : " + row[2] + "\n"
        
    elif(action == "delete_conv"):
        db.deleteByUserName(userid)
        answer = "대화내역이 모두 삭제되었습니다."

    message_attachments = [
            {
                "text": answer,
                "callback_id": "answer",
                "color": "#EBB424",
                "attachment_type": "default",
            }
        ]

    slack_web_client.chat_postMessage(
            channel=slack_payload["channel"].get("id"),
            attachments=json.dumps(message_attachments)
    )

    return make_response("", 200)

@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)
