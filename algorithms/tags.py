from urllib import request
from bs4 import BeautifulSoup
from collections import Counter

tag_list = []


def get_tags(gid: str):
    url = 'https://steamdb.info/app/%s/info/' % gid
    headers = {'user-agent': 'Chrome/54.0.2840.59'}
    req = request.Request(url=url, headers=headers, method='GET')
    page = request.urlopen(req).read().decode('utf-8')
    soup = BeautifulSoup(page, 'html.parser')

    for kind in soup.find_all('td', class_='span3'):
        if kind.string == 'store_tags':
            tags = kind.find_next_sibling('td')
            for tag in tags.find_all('a'):
                tag_list.append(tag.string)
            break


if __name__ == '__main__':
    with open('lstm_rec300.txt', 'r', encoding='utf-8') as f:
        for line in f:
            game_id = line.split()[0]
            get_tags(game_id)

    for k, v in Counter(tag_list).items():
        print('%s\t%d' % (k, v))
