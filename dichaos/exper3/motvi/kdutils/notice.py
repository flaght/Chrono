import os, requests, json
def feishu(message):
    url = 'https://open.feishu.cn/open-apis/bot/v2/hook/{0}'.format(os.environ['FS_TOKEN'])
    HEADERS = {"Content-Type": "application/json ;charset=utf-8 "}
    message = message
    String_textMsg = {"msg_type": "text", "content": {"text": message}}
    String_textMsg = json.dumps(String_textMsg)
    print(String_textMsg)
    res = requests.post(url, data=String_textMsg, headers=HEADERS)
    print(res.text)