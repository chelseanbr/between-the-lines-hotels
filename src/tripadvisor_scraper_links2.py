import requests             
from bs4 import BeautifulSoup 
import csv                  
import webbrowser
import io
import re
import sys
import pandas as pd

def display(content, filename='output.html'):
    with open(filename, 'wb') as f:
        f.write(content)
        webbrowser.open(filename)

def get_soup(session, url, show=False):
    r = session.get(url)
    if show:
        display(r.content, 'temp.html')

    if r.status_code != 200: # not OK
        print('[get_soup] status code:', r.status_code)
    else:
        return BeautifulSoup(r.text, 'html.parser')
    
def post_soup(session, url, params, show=False):
    '''Read HTML from server and convert to Soup'''

    r = session.post(url, data=params)
    
    if show:
        display(r.content, 'temp.html')

    if r.status_code != 200: # not OK
        print('[post_soup] status code:', r.status_code)
    else:
        return BeautifulSoup(r.text, 'html.parser')
    
def scrape(url, lang='ALL'):

    # create session to keep all cookies (etc.) between requests
    session = requests.Session()

    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0',
    })


    items = parse(session, url + '?filterLang=' + lang)

    return items

def parse(session, url):
    '''Get number of reviews and start getting subpages with reviews'''

    print('[parse] url:', url)

    soup = get_soup(session, url)

    if not soup:
        print('[parse] no soup:', url)
        return

    # num_reviews = soup.find('span', class_='reviews_header_count').text # get text
    num_reviews = soup.find('span', class_='hotels-hotel-review-about-with-photos-Reviews__seeAllReviews--3PpLR').text # get text
    # num_reviews = num_reviews[1:-1] 
    num_reviews = num_reviews.replace(',', '')
    num_reviews = num_reviews.replace('reviews', '')
    num_reviews = int(num_reviews) # convert text into integer
    print('[parse] num_reviews ALL:', num_reviews)

    url_template = url.replace('.html', '-or{}.html')
    # print('[parse] url_template:', url_template)

    items = []

    offset = 0

    while(True):
        subpage_url = url_template.format(offset)

        subpage_items = parse_reviews(session, subpage_url)
        if not subpage_items:
            break

        items += subpage_items

        if len(subpage_items) < 5:
            break

        offset += 5

    return items

def get_reviews_ids(soup):

    items = soup.find_all('div', attrs={'data-reviewid': True})

    if items:
        reviews_ids = [x.attrs['data-reviewid'] for x in items][::2] # mini test
        # reviews_ids = [x.attrs['data-reviewid'] for x in items][0:10000:1] # get 10,000 reviews

        # reviews_ids = [x.attrs['data-reviewid'] for x in items][::1] # get all reviews
        print('[get_reviews_ids] data-reviewid:', reviews_ids)
        return reviews_ids
    
def get_more(session, reviews_ids):

    url = 'https://www.tripadvisor.com/OverlayWidgetAjax?Mode=EXPANDED_HOTEL_REVIEWS_RESP&metaReferer=Hotel_Review'

    payload = {
        'reviews': ','.join(reviews_ids), # ie. "577882734,577547902,577300887",
        #'contextChoice': 'DETAIL_HR', # ???
        'widgetChoice': 'EXPANDED_HOTEL_REVIEW_HSX', # ???
        'haveJses': 'earlyRequireDefine,amdearly,global_error,long_lived_global,apg-Hotel_Review,apg-Hotel_Review-in,bootstrap,desktop-rooms-guests-dust-en_US,responsive-calendar-templates-dust-en_US,taevents',
        'haveCsses': 'apg-Hotel_Review-in',
        'Action': 'install',
    }

    soup = post_soup(session, url, payload)

    return soup

def parse_reviews(session, url):
    '''Get all reviews from one page'''

    print('[parse_reviews] url:', url)

    soup =  get_soup(session, url)

    if not soup:
        print('[parse_reviews] no soup:', url)
        return

    hotel_name = soup.find('h1', id='HEADING').text

    reviews_ids = get_reviews_ids(soup)
    if not reviews_ids:
        return

    soup = get_more(session, reviews_ids)

    if not soup:
        print('[parse_reviews] no soup:', url)
        return

    items = []
    
#     print(soup) # DEBUG
    
    for idx, review in enumerate(soup.find_all('div', class_='reviewSelector')):
        try:
            badgets = review.find_all('span', class_='badgetext')
            # print(badgets)
            if len(badgets) > 0:
                contributions = badgets[0].text
            else:
                contributions = '0'

            if len(badgets) > 1:
                helpful_vote = badgets[1].text
            else:
                helpful_vote = '0'
            user_loc = review.select_one('div.userLoc strong')
            if user_loc:
                user_loc = user_loc.text
            else:
                user_loc = ''
                
            bubble_rating = review.select_one('span.ui_bubble_rating')['class']
            bubble_rating = int(bubble_rating[1].split('_')[-1])/10
            # print(bubble_rating)

            review_id = reviews_ids[idx]

            username_string = str(review.find('div', class_='info_text pointer_cursor'))
#             print(type(username_string))
#             print(type(str(username_string)))
            username_result = re.search("<div>(.*)</div></div>", username_string)
            user_name = username_result.group(1)
            # print(username_string)
            # print(user_name) #DEBUG
            
            item = {
                'hotel_name': hotel_name,
                'review_body': review.find('p', class_='partial_entry').text,
                'review_date': review.find('span', class_='ratingDate')['title'], # 'ratingDate' instead of 'relativeDate'

#                 'user_name':'',
                'user_name': user_name,
                
                'rating': bubble_rating,
                # 'contributions': contributions,
                'helpful_vote': helpful_vote,
                'user_location': user_loc,
                'review_id': review_id,
                'url': url
            }

            items.append(item)
            # print('\n--- review ---\n')
            # for key,val in item.items():
            #     print(' ', key, ':', val)

        except:
            print("Something went wrong with review #" + review_id)
            continue

    print()

    return items

def write_in_csv(items, filename='results.csv',
                  headers=['review id', 'hotel name', 'review title', 'review body',
                           'review date', 'contributions', 'helpful vote',
                           'user name' , 'user location', 'rating', 'url'],
                  mode='w'):

    print('--- CSV ---')

    with io.open(filename, mode, encoding="utf-8") as csvfile:
        csv_file = csv.DictWriter(csvfile, headers)

        if mode == 'w':
            csv_file.writeheader()

        csv_file.writerows(items)

def main(start_urls, pg):
    DB_COLUMN0  = 'review_id'
    DB_COLUMN1  = 'url'
    DB_COLUMN2 = 'hotel_name'
    DB_COLUMN3 = 'review_date'
    DB_COLUMN4 = 'review_body'
    DB_COLUMN5 = 'user_location'
    # DB_COLUMN6 = 'contributions'
    DB_COLUMN6 = 'user_name'
    DB_COLUMN7 = 'helpful_vote'
    DB_COLUMN8 = 'rating'

    start_urls = start_urls

    lang = 'en'

    headers = [ 
        DB_COLUMN0,
        DB_COLUMN1, 
        DB_COLUMN2, 
        DB_COLUMN3,
        DB_COLUMN4,
        DB_COLUMN5,
        DB_COLUMN6,
        DB_COLUMN7,
        DB_COLUMN8,
    ]

    for url in start_urls:
        try:
            # get all reviews for 'url' and 'lang'
            items = scrape(url, lang)

            if not items:
                print('No reviews')
            else:
                # write in CSV
                filename = url.split('Reviews-')[1][:-5] + '__' + lang
                print('filename:', filename)

                # file_dir = '../data/web_scraped/'
                # file_dir = 'data/web_scraped/aws/' + pg + '/'
                file_dir = 'data/' + pg + '/'
                # file_dir = 'data/web_scraped/aws_test/' + pg +'/'

                write_in_csv(items, file_dir + filename + '.csv', headers, mode='w')
        except:
            print("Something went wrong with " + url)
            continue


if __name__ == "__main__":
    # csv_arg = sys.argv[-1]
    # print(csv_arg)
    # if '.csv' not in csv_arg:
    #     print('Please provide argument for filepath/name of csv with Tripadvisor links to scrape.')
    #     sys.exit()
    # else:
    #     csv = csv_arg

    # csv = 'links_1.csv'
    # links_df = pd.read_csv(csv)
    # # print(links_df)
    # # print(list(links_df['url']))
    # links = list(links_df['url'])

    # test = links

    links_2 = ['https://www.tripadvisor.com/Hotel_Review-g187147-d12455412-Reviews-CitizenM_Paris_Gare_de_Lyon-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d250927-Reviews-Hotel_Europe_Saint_Severin-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d197424-Reviews-Novotel_Paris_Les_Halles-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d6395273-Reviews-Hotel_Eiffel_Blomet-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d233514-Reviews-Hotel_Mademoiselle-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d582530-Reviews-Kube_Hotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d597573-Reviews-Novotel_Paris_Centre_Gare_Montparnasse-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d194232-Reviews-Hotel_Eiffel_Turenne-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d595754-Reviews-Hyatt_Regency_Paris_Etoile-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d233573-Reviews-Hotel_Balmoral-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d207663-Reviews-K_K_Hotel_Cayre-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d271839-Reviews-Hotel_Muguet-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d197563-Reviews-Hotel_La_Manufacture-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d274978-Reviews-Hotel_Darcet-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d229577-Reviews-Villa_Pantheon-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d1633522-Reviews-Hotel_Design_Secret_de_Paris-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d9566832-Reviews-Hotel_34B_Astotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d197985-Reviews-Pullman_Paris_Tour_Eiffel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d233583-Reviews-Hotel_R_de_Paris-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d282205-Reviews-Hotel_Terminus_Lyon-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d197656-Reviews-Mercure_Paris_Centre_Eiffel_Tower_Hotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d280083-Reviews-B_Montmartre_Hotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d4340356-Reviews-Hotel_Fabric-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d198085-Reviews-Terrass_Hotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d220006-Reviews-Grand_Hotel_Leveque-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d228694-Reviews-Hotel_Malte_Astotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d197946-Reviews-Hotel_Bradford_Elysees_Astotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d194276-Reviews-Hotel_Dauphine_Saint_Germain-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d219977-Reviews-Relais_Christine-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d194284-Reviews-Hotel_Sainte_Beuve-Paris_Ile_de_France.html']
    
    links_3 = ['',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '']
    
    links_4 = ['',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '',\
                '']

    # print(type(test))
    # test = ['https://www.tripadvisor.com/Hotel_Review-g60763-d7816364-Reviews-Executive_Hotel_LeSoleil-New_York_City_New_York.html',\
    #     'https://www.tripadvisor.com/Hotel_Review-g60763-d93437-Reviews-Hotel_Edison-New_York_City_New_York.html',\
    #     'https://www.tripadvisor.com/Hotel_Review-g60763-d3533197-Reviews-Hyatt_Union_Square_New_York-New_York_City_New_York.html']

    url_pgs = [links_2]
    pgs = ['links_2']

    for url_pg, pg in zip(url_pgs, pgs):
        print(pg)
        try:
            main(url_pg, pg)
        except:
            print("Something went wrong with " + pg)
            continue