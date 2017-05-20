import time


# 尝试获取并返回令牌
def check_token(conn, token):
    return conn.hget('login:', token)


# 记录用户最近浏览的25个商品
def update_token(conn, token, user, item=None):
    timestamp = time.time()
    conn.hset('login:', token, user)
    conn.zadd('recent:', token, timestamp)
    if item:
        conn.zadd('viewed:' + token, item, timestamp)
        conn.zremrangebyrank('viewed:' + token, 0, -26)


# 清理用户登陆session
