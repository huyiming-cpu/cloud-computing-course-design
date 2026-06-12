import os
import redis
from flask import Flask, jsonify

app = Flask(__name__)

# 从 ConfigMap（envFrom）和 Secret（secretKeyRef）读取配置
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', '6379'))
redis_password = os.getenv('REDIS_PASSWORD', '')

# 全局变量，支持运行时更新
redis_ok = False
r = None

def init_redis():
    global r, redis_ok
    try:
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password if redis_password else None,
            decode_responses=True
        )
        r.ping()
        redis_ok = True
    except Exception as e:
        redis_ok = False

init_redis()

@app.route('/api/ping')
def ping():
    return jsonify({"status": "ok"})

@app.route('/api/redis')
def redis_status():
    return jsonify({
        "redis_connected": redis_ok,
        "redis_host": redis_host,
        "redis_port": redis_port
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)