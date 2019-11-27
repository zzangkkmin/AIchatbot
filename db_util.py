import sqlite3
from datetime import datetime

def init():
    conn = getConnection()
    c = conn.cursor()
    c.execute("DROP TABLE HISTORY_CONVERSATION_TB")
    c.execute("CREATE TABLE HISTORY_CONVERSATION_TB (ID INTEGER PRIMARY KEY AUTOINCREMENT, QUESTION TEXT, ANSWER TEXT, USER_NM TEXT, REG_DT TEXT)")
    close(conn)

def getConnection():
    conn = sqlite3.connect('app.db')
    return conn

def close(conn):
    conn.commit()
    conn.close()
    
def insert(question, answer, user_name):
    conn = getConnection()
    c = conn.cursor()
    c.execute("INSERT INTO HISTORY_CONVERSATION_TB (QUESTION, ANSWER, USER_NM, REG_DT) VALUES (?, ?, ?, ?);",
             (question, answer, user_name, datetime.now()))
    close(conn)
    return c.lastrowid

def update(id, answer):
    conn = getConnection()
    c = conn.cursor()
    c.execute("UPDATE HISTORY_CONVERSATION_TB SET ANSWER = (?), REG_DT = (?) WHERE ID = (?);", (answer, datetime.now(), id))
    close(conn)
    return c.lastrowid

def deleteByUserName(username):
    conn = getConnection()
    c = conn.cursor()
    c.execute("DELETE FROM HISTORY_CONVERSATION_TB WHERE USER_NM = (?);", (username,))
    close(conn)

def findById(id):
    conn = getConnection()
    c = conn.cursor()
    c.execute("SELECT * FROM HISTORY_CONVERSATION_TB WHERE ID = (?);", (id,))
    result = c.fetchall()
    close(conn)
    return result

def findByUserName(user_name):
    conn = getConnection()
    c = conn.cursor()
    c.execute("SELECT * FROM HISTORY_CONVERSATION_TB WHERE USER_NM = (?);", (user_name,))
    result = c.fetchall()
    close(conn)
    return result


def findByAnswer(question):
    conn = getConnection()
    c = conn.cursor()
    c.execute("SELECT * FROM HISTORY_CONVERSATION_TB WHERE QUESTION LIKE (?);", (question,))
    result = c.fetchall()
    close(conn)
    return result