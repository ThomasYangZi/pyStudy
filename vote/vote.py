import time
import unittest


# 定义常量
ONE_WEEK_IN_SECONDS = 7 * 84600
VOTE_SCORE = 432
ARTICLES_PRE_PAGE = 10


# 投票函数
def article_vote(conn, user, article):
    # 计算投票的截止时间
    cutoff = time.time() - ONE_WEEK_IN_SECONDS
    # 判断是否已经截止
    if conn.zscore('time:', article) < cutoff:
        return

    # 获取文章id
    article_id = article.partition(':')[-1]
    # 更新文章的分值和更新文章的投票数
    if conn.sadd('voted:' + article_id, user):
        conn.zincrby('score:', article, VOTE_SCORE)
        conn.hincrby(article, 'votes', 1)


# 发布和获取文章
def post_article(conn, user, title, link):
    # 从'article:'中获取一个自增的id，并获取这个id
    article_id = str(conn.incr('article:'))

    # 将文章发布者添加到已投票用户列表中
    # 设置用户列表的过期时间
    voted = 'voted:' + article_id
    conn.sadd(voted, user)
    conn.expire(voted, ONE_WEEK_IN_SECONDS)

    # 将文章写入Hash散列表中
    now = time.time()
    article = 'article:' + article_id
    conn.hmset(article, {
        'title': title,
        'link': link,
        'poster': user,
        'time': now,
        'votes': 1
    })

    conn.zadd('score:', article, now + VOTE_SCORE)
    conn.zadd('time:', article, now)

    return article_id


# 获取文章列表并排序
def get_articles(conn, page, order='score:'):
    start = (page - 1) * ARTICLES_PRE_PAGE
    end = start + ARTICLES_PRE_PAGE - 1

    ids = conn.zrevrange(order, start, end)
    articles = []
    for id in ids:
        article_data = conn.hgetall(id)

        article_data['id'] = id
        articles.append(article_data)

    return articles


# 添加群组功能
def add_remove_groups(conn, article_id, to_add=None, to_remove=None):
    if to_remove is None:
        to_remove = []
    if to_add is None:
        to_add = []
    article = 'article:' + article_id
    for group in to_add:
        conn.sadd('group:' + group, article)
    for group in to_remove:
        conn.srem('group:' + group, article)


# 获取群组文章并排序
def get_group_articles(conn, group, page, order='score:'):
    key = order + group
    if not conn.exists(key):
        conn.zinterstore(key, ['group:' + group, order], aggregate='max')
        conn.expire(key, 60)
    return get_articles(conn, page, key)


#--------------- 以下是用于测试代码的辅助函数 --------------------------------

class TestCh01(unittest.TestCase):
    def setUp(self):
        import redis
        self.conn = redis.Redis(db=15)

    def tearDown(self):
        del self.conn
        print()
        print()

    def test_article_functionality(self):
        conn = self.conn
        import pprint

        article_id = str(post_article(conn, 'username', 'A title', 'http://www.google.com'))
        print("We posted a new article with id:", article_id)
        print()
        self.assertTrue(article_id)

        print("Its HASH looks like:")
        r = conn.hgetall('article:' + article_id)
        print(r)
        print()
        self.assertTrue(r)

        article_vote(conn, 'other_user', 'article:' + article_id)
        print("We voted for the article, it now has votes:")
        v = int(conn.hget('article:' + article_id, 'votes'))
        print(v)
        print()
        self.assertTrue(v > 1)

        print("The currently highest-scoring articles are:")
        articles = get_articles(conn, 1)
        pprint.pprint(articles)
        print()

        self.assertTrue(len(articles) >= 1)

        add_remove_groups(conn, article_id, ['new-group'])
        print("We added the article to a new group, other articles include:")
        articles = get_group_articles(conn, 'new-group', 1)
        pprint.pprint(articles)
        print()
        self.assertTrue(len(articles) >= 1)

        to_del = (
            conn.keys('time:*') + conn.keys('voted:*') + conn.keys('score:*') +
            conn.keys('article:*') + conn.keys('group:*')
        )
        if to_del:
            conn.delete(*to_del)

if __name__ == '__main__':
    unittest.main()


