import json
import re


def parse_messages_to_str(messages):
    res = ""
    for msg in messages:
        res += "role={}\n".format(msg.get("role"))
        res += msg.get("content")
    return res


def parse_response_to_str(response):
    return response.get("content")


def parse_json_response(response_str):
    pattern = r"```json\s*(.*?)```"
    match = re.search(pattern, response_str, re.DOTALL | re.IGNORECASE)
    
    if match:
        text = match.group(1).strip()
    else:
        text = response_str.strip()
    
    return json.loads(text)
