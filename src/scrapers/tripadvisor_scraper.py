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
        # reviews_ids = [x.attrs['data-reviewid'] for x in items][::2] # mini test
        # reviews_ids = [x.attrs['data-reviewid'] for x in items][0:10000:1] # get 10,000 reviews

        reviews_ids = [x.attrs['data-reviewid'] for x in items][::1] # get all reviews
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

    links_1 = ['https://www.tripadvisor.com/Hotel_Review-g60763-d1949647-Reviews-Fairfield_Inn_Suites_New_York_Manhattan_Chelsea-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d4232686-Reviews-Le_Meridien_New_York_Central_Park-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d2622936-Reviews-Wyndham_Garden_Chinatown-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93450-Reviews-Grand_Hyatt_New_York-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d75737-Reviews-Night_Theater_District-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d19271048-Reviews-NY_Finest_Luxury_Apartment-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93466-Reviews-Double_Tree_by_Hilton_Hotel_Metropolitan_New_York_City-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d249656-Reviews-Millennium_Premier_New_York_Times_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1746459-Reviews-The_Nolitan_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d12076837-Reviews-Hotel_Henri-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d11998087-Reviews-Embassy_Suites_by_Hilton_New_York_Manhattan_Times_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93359-Reviews-Econo_Lodge_Times_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d2151571-Reviews-Selina_Chelsea-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1806062-Reviews-Cassa_Hotel_NY_45th_Street-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d217627-Reviews-Belnord_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d113317-Reviews-Casablanca_Hotel_by_Library_Hotel_Collection-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1858923-Reviews-DoubleTree_by_Hilton_New_York_Downtown-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d4283443-Reviews-The_High_Line_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d14095381-Reviews-Moxy_NYC_Downtown-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93383-Reviews-6_Columbus_Central_Park_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1785018-Reviews-The_James_New_York_SoHo-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d223023-Reviews-Hudson_New_York-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d112021-Reviews-NYCASA_46_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93396-Reviews-The_Iroquois_New_York-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93419-Reviews-The_Carlyle_A_Rosewood_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d209382-Reviews-Millennium_Hilton_New_York_One_UN_Plaza-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d122014-Reviews-Gild_Hall_A_Thompson_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d217630-Reviews-414_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d214197-Reviews-Hotel_Pennsylvania-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d3572583-Reviews-Holiday_Inn_Express_Manhattan_Midtown_West-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d3522780-Reviews-Walker_Hotel_Greenwich_Village-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d671150-Reviews-The_Empire_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g14129477-d7371377-Reviews-Aman_Tokyo-Otemachi_Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066444-d585204-Reviews-Mandarin_Oriental_Tokyo-Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129730-d579136-Reviews-The_Prince_Park_Tower_Tokyo-Shibakoen_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066443-d17667937-Reviews-Reyado_Hotel_Kudan-Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066456-d630480-Reviews-Shibuya_Granbell_hotel-Shibuya_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066451-d10234426-Reviews-Hotel_Allamanda_Aoyama-Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066461-d19392762-Reviews-Prostyle_Ryokan_Tokyo_Asakusa-Taito_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14133659-d1486443-Reviews-Hotel_Keihan_Tokyo_Yotsuya-Yotsuya_Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14133667-d4994810-Reviews-Shinjuku_Granbell_Hotel-Kabukicho_Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129743-d300459-Reviews-ANA_InterContinental_Tokyo-Akasaka_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129730-d301389-Reviews-Tokyo_Prince_Hotel-Shibakoen_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129477-d12158513-Reviews-Ascott_Marunouchi_Tokyo-Otemachi_Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066451-d571809-Reviews-Conrad_Tokyo-Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066443-d3676666-Reviews-Hotel_New_Otani_Garden_Tower-Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129743-d634263-Reviews-The_Ritz_Carlton_Tokyo-Akasaka_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129528-d310308-Reviews-The_Tokyo_Station_Hotel-Marunouchi_Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066854-d498905-Reviews-Tokyo_Marriott_Hotel-Shinagawa_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129528-d301911-Reviews-Four_Seasons_Hotel_Tokyo_at_Marunouchi-Marunouchi_Chiyoda_Tokyo_Tokyo_Prefecture_Kant.html', 'https://www.tripadvisor.com/Hotel_Review-g1066451-d307378-Reviews-The_Royal_Park_Hotel_Iconic_Tokyo_Shiodome-Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066456-d14212238-Reviews-Shibuya_Stream_Excel_Hotel_Tokyu-Shibuya_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066444-d19501481-Reviews-Hotel_K5-Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066458-d10082928-Reviews-BnA_HOTEL_Koenji-Suginami_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066444-d14132415-Reviews-Mitsui_Garden_Hotel_Nihonbashi_Premier-Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129647-d6485175-Reviews-Andaz_Tokyo_Toranomon_Hills-Toranomon_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066454-d2703580-Reviews-First_Cabin_Haneda_Terminal_1-Ota_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14131019-d301356-Reviews-Grand_Prince_Hotel_New_Takanawa-Takanawa_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129536-d580687-Reviews-The_Peninsula_Tokyo-Yurakucho_Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066456-d479494-Reviews-Shibuya_Tokyu_REI_Hotel-Shibuya_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129734-d18854227-Reviews-The_Lively_Azabujuban_Tokyo-Azabujuban_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129735-d306610-Reviews-Grand_Hyatt_Tokyo-Roppongi_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d214685-Reviews-OHANA_Waikiki_Malia_by_Outrigger-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.ca/Hotel_Review-g60982-d87016-Reviews-Hilton_Hawaiian_Village_Waikiki_Beach_Resort-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d2555768-Reviews-Shoreline_Hotel_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87035-Reviews-Manoa_Valley_Inn-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d114027-Reviews-Ambassador_Hotel_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86978-Reviews-Aston_Waikiki_Beach_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86954-Reviews-The_Polynesian_Residences_Hotel_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d114010-Reviews-Aston_at_the_Executive_Centre_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86984-Reviews-Aston_Waikiki_Circle_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87015-Reviews-Waikiki_Monarch_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d114024-Reviews-The_Royal_Hawaiian_a_Luxury_Collection_Resort_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d114026-Reviews-Halepuna_Waikiki_by_Halekulani-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d90017-Reviews-DoubleTree_by_Hilton_Alana_Waikiki_Beach-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87149-Reviews-Waikiki_Sand_Villa_Hotel-Honolulu_Oahu_Hawaii.html', \
        'https://www.tripadvisor.com/Hotel_Review-g60982-d219292-Reviews-Courtyard_by_Marriott_Waikiki_Beach-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d12077161-Reviews-Holiday_Inn_Express_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87025-Reviews-The_Imperial_Hawaii_Resort-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86969-Reviews-The_Laylow_Autograph_Collection-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d559726-Reviews-Aqua_Palms_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d209422-Reviews-Hilton_Waikiki_Beach-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d4020497-Reviews-Vive_Hotel_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86951-Reviews-The_New_Otani_Kaimana_Beach_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86983-Reviews-Espacio_The_Jewel_of_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86957-Reviews-Ala_Moana_Honolulu_by_Mantra-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d224616-Reviews-Hilton_Grand_Vacations_at_Hilton_Hawaiian_Village-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87104-Reviews-Sheraton_Princess_Kaiulani-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86961-Reviews-Aston_at_the_Waikiki_Banyan-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d114031-Reviews-Waikiki_Beachcomber_by_Outrigger-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87008-Reviews-The_Equus-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87018-Reviews-Ramada_Plaza_by_Wyndham_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d3973287-Reviews-Best_Western_Premier_Herald_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d75711-Reviews-The_Gallivant_Times_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1646128-Reviews-InterContinental_New_York_Times_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d14149815-Reviews-Moxy_NYC_Chelsea-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93520-Reviews-Park_Central_Hotel_New_York-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d7891458-Reviews-Arlo_SoHo-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93579-Reviews-Park_Lane_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d12301470-Reviews-Moxy_NYC_Times_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d80112-Reviews-San_Carlos_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d99288-Reviews-Arthouse_Hotel_New_York_City-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d122005-Reviews-The_New_Yorker_a_Wyndham_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d2079052-Reviews-YOTEL_New_York-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d4799063-Reviews-Hyatt_Centric_Times_Square_New_York-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d4451787-Reviews-Club_Quarters_Hotel_Grand_Central-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d14149780-Reviews-Aliz_Hotel_Times_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d15135187-Reviews-Park_Terrace_Hotel_on_Bryant_Park-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d99352-Reviews-Hilton_Garden_Inn_Times_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93454-Reviews-Crowne_Plaza_Times_Square_Manhattan-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d10541730-Reviews-Arlo_NoMad-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d8035866-Reviews-Hampton_Inn_Manhattan_Times_Square_Central-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d8515751-Reviews-Hotel_Riu_Plaza_New_York_Times_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d99766-Reviews-The_Roosevelt_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93555-Reviews-Sheraton_New_York_Times_Square_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d5040757-Reviews-Homewood_Suites_by_Hilton_New_York_Midtown_Manhattan_Times_Square_South_NY-New_York_Cit.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d7816364-Reviews-Executive_Hotel_LeSoleil-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93562-Reviews-Stewart_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93507-Reviews-New_York_Marriott_Marquis-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d2554351-Reviews-Hyatt_Place_New_York_Midtown_South-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1218720-Reviews-The_Standard_High_Line-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93437-Reviews-Hotel_Edison-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d611947-Reviews-New_York_Hilton_Midtown-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g1066450-d301789-Reviews-The_Westin_Tokyo-Meguro_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14131019-d594629-Reviews-Grand_Prince_Hotel_Takanawa-Takanawa_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066442-d307334-Reviews-Hotel_Chinzanso_Tokyo-Bunkyo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066459-d12817652-Reviews-Moxy_Tokyo_Kinshicho-Sumida_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129477-d1082523-Reviews-KKR_Hotel_Tokyo-Otemachi_Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066443-d10031117-Reviews-The_Prince_Gallery_Tokyo_Kioicho_A_Luxury_Collection_Hotel-Chiyoda_Tokyo_Tokyo_Prefe.html', 'https://www.tripadvisor.com/Hotel_Review-g1066456-d307381-Reviews-Shibuya_Excel_Hotel_Tokyu-Shibuya_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066457-d17612771-Reviews-Mitsui_Garden_Hotel_Jingugaien_Tokyo_Premier-Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066451-d301657-Reviews-Park_Hotel_Tokyo-Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066451-d15662747-Reviews-Albida_Hotel_Aoyama-Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14133713-d320581-Reviews-Hotel_Sunroute_Plaza_Shinjuku-Yoyogi_Shibuya_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129573-d503002-Reviews-Mercure_Tokyo_Ginza-Ginza_Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066461-d3173252-Reviews-THE_GATE_HOTEL_Asakusa_Kaminarimon_by_HULIC-Taito_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14133673-d304289-Reviews-Hilton_Tokyo-Nishishinjuku_Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129578-d17353295-Reviews-THE_BLOSSOM_HIBIYA-Shimbashi_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129520-d1479297-Reviews-Hotel_Ryumeikan_Tokyo-Yaesu_Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14133673-d1766584-Reviews-Tokyu_Stay_Nishi_Shinjuku-Nishishinjuku_Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14133667-d1070779-Reviews-Citadines_Central_Shinjuku_Tokyo-Kabukicho_Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14134368-d302387-Reviews-Grand_Nikko_Tokyo_Daiba-Daiba_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129573-d7002465-Reviews-Millennium_Mitsui_Garden_Hotel_Tokyo-Ginza_Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129528-d2528953-Reviews-Palace_Hotel_Tokyo-Marunouchi_Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129573-d299764-Reviews-Courtyard_by_Marriott_Tokyo_Ginza_Hotel-Ginza_Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066456-d506006-Reviews-Shibuya_Tobu_Hotel-Shibuya_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066454-d6777496-Reviews-The_Royal_Park_Hotel_Tokyo_Haneda-Ota_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14133673-d304305-Reviews-Keio_Plaza_Hotel_Tokyo-Nishishinjuku_Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129528-d1235856-Reviews-Shangri_La_Hotel_Tokyo-Marunouchi_Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066451-d1115512-Reviews-Tokyu_Stay_Aoyama_Premier-Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066442-d307395-Reviews-Tokyo_Dome_Hotel-Bunkyo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066443-d301279-Reviews-Hotel_New_Otani_Tokyo_The_Main-Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', \
            'https://www.tripadvisor.com/Hotel_Review-g14129573-d2232760-Reviews-Solaria_nishitetsu_hotel_Ginza-Ginza_Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066456-d301250-Reviews-Cerulean_Tower_Tokyu_Hotel-Shibuya_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066444-d307375-Reviews-Royal_Park_Hotel-Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129743-d10223698-Reviews-HOTEL_MYSTAYS_PREMIER_Akasaka-Akasaka_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066444-d12922995-Reviews-Hotel_Intergate_Tokyo_Kyobashi-Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14133673-d15564204-Reviews-Daiwa_Roynet_Hotel_Nishi_Shinjuku-Nishishinjuku_Shinjuku_Tokyo_Tokyo_Prefecture_Kan.html', 'https://www.tripadvisor.com/Hotel_Review-g1066454-d12900097-Reviews-Relief_Premium_Haneda_Airport-Ota_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129573-d526766-Reviews-Hotel_Monteray_Ginza-Ginza_Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066459-d12964918-Reviews-Sotetsu_Fresa_Inn_Tokyo_Kinshicho-Sumida_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066449-d301686-Reviews-Hotel_East_21_Tokyo-Koto_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066454-d6894889-Reviews-Hotel_Mystays_Haneda-Ota_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066459-d1083549-Reviews-Tobu_Hotel_Levant_Tokyo-Sumida_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066443-d304288-Reviews-Hotel_Grand_Arc_Hanzomon-Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129743-d4544518-Reviews-Centurion_Hotel_Grand_Akasaka-Akasaka_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129608-d1105294-Reviews-Hotel_Sardonyx_Tokyo-Hatchobori_Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14133667-d8557280-Reviews-APA_Hotel_Shinjuku_Kabukicho_Tower-Kabukicho_Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14129727-d12982713-Reviews-Henn_na_Hotel_Tokyo_Hamamatsucho-Hamamatsucho_Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066451-d479492-Reviews-Hotel_The_Celestine_Tokyo_Shiba-Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066456-d1083444-Reviews-Sakura_Hotel_Hatagaya-Shibuya_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066443-d307319-Reviews-Akasaka_Excel_Hotel_Tokyu-Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066457-d12123803-Reviews-Nine_hours_Shinjuku_North-Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066460-d301382-Reviews-Sunshine_City_Prince_Hotel-Toshima_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14134362-d15083542-Reviews-Daiwa_Roynet_Hotel_Tokyo_Ariake-Ariake_Koto_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066457-d6857433-Reviews-APA_Hotel_Shinjuku_Gyoemmae-Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066456-d13130379-Reviews-APA_Hotel_Resort_Nishishinjuku_Gochome_eki_Tower-Shibuya_Tokyo_Tokyo_Prefecture_Kant.html', 'https://www.tripadvisor.com/Hotel_Review-g1066444-d15668612-Reviews-Hamacho_Hotel_Tokyo-Chuo_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g14133673-d1090409-Reviews-Nishitetsu_Inn_Shinjuku-Nishishinjuku_Shinjuku_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066443-d304296-Reviews-Sakura_Hotel_Jimbocho-Chiyoda_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066454-d12958764-Reviews-Keikyu_EX_Inn_Haneda-Ota_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g1066451-d666515-Reviews-Mitsui_Garden_Hotel_Shiodome_Italia_gai-Minato_Tokyo_Tokyo_Prefecture_Kanto.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d1199673-Reviews-Grand_Waikikian_by_Hilton_Grand_Vacations-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d79500-Reviews-Best_Western_The_Plaza_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d8423429-Reviews-Stay_Condominiums_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d90159-Reviews-Aqua_Skyline_at_Island_Colony-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d14100457-Reviews-OHANA_Waikiki_East_by_Outrigger-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d214688-Reviews-Luana_Waikiki_Hotel_Suites-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d8429471-Reviews-Aqua_Ohia_Waikiki_Studio_Suites-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87102-Reviews-Moana_Surfrider_A_Westin_Resort_Spa_Waikiki_Beach-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d90015-Reviews-Halekulani_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87011-Reviews-Prince_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87147-Reviews-Waikiki_Central_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d225459-Reviews-Aston_Waikiki_Beach_Tower-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87119-Reviews-Hotel_LaCroix-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d596760-Reviews-Hotel_Renew-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d1567958-Reviews-Kuhio_Banyan_Club-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d120684-Reviews-Outrigger_Waikiki_Beach_Resort-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d3635742-Reviews-Marina_Tower_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d13912965-Reviews-Paniolo_at_the_Equus-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87090-Reviews-Hilton_Garden_Inn_Waikiki_Beach-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d14124537-Reviews-Ilikai_Lite-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d242435-Reviews-Hyatt_Regency_Waikiki_Beach_Resort_Spa-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d17724495-Reviews-Pagoda_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d15610197-Reviews-Real_Select_Vacations_at_The_Ritz_Carlton_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86953-Reviews-Holiday_Surf_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87092-Reviews-Pagoda_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d615153-Reviews-Embassy_Suites_By_Hilton_Waikiki_Beach_Walk-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d114008-Reviews-The_Kahala_Hotel_Resort-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d120687-Reviews-Ewa_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d208960-Reviews-Castle_Waikiki_Grand_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d4604031-Reviews-Hokulani_Waikiki_by_Hilton_Grand_Vacations-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86973-Reviews-Pacific_Monarch_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d10229548-Reviews-The_Ritz_Carlton_Residences-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86988-Reviews-Aqua_Oasis-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d1818106-Reviews-The_Modern_Honolulu-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87031-Reviews-Aqua_Bamboo_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d2385826-Reviews-Stay_Hotel_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87091-Reviews-Pacific_Marina_Inn-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87095-Reviews-Queen_Kapiolani_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86966-Reviews-Coconut_Waikiki_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87111-Reviews-Alohilani_Resort_Waikiki_Beach-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87108-Reviews-Sheraton_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d10728599-Reviews-Hyatt_Centric_Waikiki_Beach-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d534764-Reviews-Waikiki_Resort_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d90108-Reviews-Lotus_Honolulu_at_Diamond_Head-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87040-Reviews-Hyatt_Place_Waikiki_Beach-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86970-Reviews-Pearl_Hotel_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d209424-Reviews-Ilikai_Hotel_Luxury_Suites-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d214683-Reviews-Waikiki_Beach_Marriott_Resort_Spa-Honolulu_Oahu_Hawaii.html', \
                'https://www.tripadvisor.com/Hotel_Review-g60982-d214680-Reviews-Park_Shore_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87024-Reviews-Ilima_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86952-Reviews-Waikiki_Beachside_Hostel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d1484551-Reviews-Trump_International_Hotel_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d2202220-Reviews-White_Sands_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d227765-Reviews-The_Surfjack_Hotel_Swim_Club-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87039-Reviews-Marine_Surf_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87052-Reviews-Outrigger_Reef_Waikiki_Beach_Resort-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d3686316-Reviews-The_Beach_Waikiki_Boutique_Hostel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87097-Reviews-Royal_Grove_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d86959-Reviews-Aqua_Aloha_Surf_Waikiki-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60982-d87017-Reviews-Airport_Honolulu_Hotel-Honolulu_Oahu_Hawaii.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d224224-Reviews-The_Ritz_Carlton_New_York_Central_Park-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d596200-Reviews-Hudson_River_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93358-Reviews-Pod_51_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d249710-Reviews-Morningside_Inn-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d7595213-Reviews-Four_Points_by_Sheraton_New_York_Downtown-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1164645-Reviews-Eurostars_Wall_Street-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d220243-Reviews-W_New_York_Union_Square-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1572980-Reviews-Crosby_Street_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93401-Reviews-Heritage_Hotel_New_York_City-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93475-Reviews-New_York_Marriott_East_Side-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1462005-Reviews-W_New_York_Downtown-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d6701149-Reviews-Midtown_West_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d113298-Reviews-Four_Seasons_Hotel_New_York-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1601249-Reviews-Hilton_New_York_Fashion_District-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93452-Reviews-LUXE_Life_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d93374-Reviews-Off_Soho_Suites-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d10846801-Reviews-The_Whitby_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1783324-Reviews-Sheraton_Tribeca_New_York_Hotel-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d1500405-Reviews-Hampton_Inn_Manhattan_Times_Square_South-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d113311-Reviews-The_Peninsula_New_York-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d281071-Reviews-Mandarin_Oriental_New_York-New_York_City_New_York.html', 'https://www.tripadvisor.com/Hotel_Review-g60763-d3533197-Reviews-Hyatt_Union_Square_New_York-New_York_City_New_York.html']
    
    # print(type(test))
    # test = ['https://www.tripadvisor.com/Hotel_Review-g60763-d7816364-Reviews-Executive_Hotel_LeSoleil-New_York_City_New_York.html',\
    #     'https://www.tripadvisor.com/Hotel_Review-g60763-d93437-Reviews-Hotel_Edison-New_York_City_New_York.html',\
    #     'https://www.tripadvisor.com/Hotel_Review-g60763-d3533197-Reviews-Hyatt_Union_Square_New_York-New_York_City_New_York.html']

    url_pgs = [links_1]
    pgs = ['links_1']

    for url_pg, pg in zip(url_pgs, pgs):
        print(pg)
        try:
            main(url_pg, pg)
        except:
            print("Something went wrong with " + pg)
            continue