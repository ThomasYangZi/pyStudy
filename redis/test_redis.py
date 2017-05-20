import redis

conn = redis.Redis()
conn.set('hello', 'world')

# True

print(conn.get('hello'))

