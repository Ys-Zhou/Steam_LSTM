import urllib.request
from bs4 import BeautifulSoup

tag_list = []


def get_tags(gid: str):
    url = 'https://steamdb.info/app/%s/info/' % gid
    headers = {'user-agent': 'Chrome/54.0.2840.59'}
    req = urllib.request.Request(url=url, headers=headers, method='GET')
    page = urllib.request.urlopen(req).read().decode('utf-8')
    soup = BeautifulSoup(page, 'html.parser')

    for kind in soup.find_all('td', class_='span3'):
        if kind.string == 'store_tags':
            tags = kind.find_next_sibling('td')
            for tag in tags.find_all('a'):
                tag_list.append(tag.string)
            break


if __name__ == '__main__':
    get_tags('578080')
    print(tag_list)
