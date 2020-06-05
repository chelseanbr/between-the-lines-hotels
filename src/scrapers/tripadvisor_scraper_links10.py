import requests             
from bs4 import BeautifulSoup 
import csv                  
import webbrowser
import io
import re
import sys
# import pandas as pd

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
                'https://www.tripadvisor.com/Hotel_Review-g187147-d229601-Reviews-Maison_FL-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d254597-Reviews-Best_Western_Plus_61_Paris_Nation_Hotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d250928-Reviews-Park_Hyatt_Paris_Vendome-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d194276-Reviews-Hotel_Dauphine_Saint_Germain-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d207854-Reviews-Villa_Beaumarchais-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d6675948-Reviews-Hotel_Da_Vinci_Spa-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d207566-Reviews-Hotel_Le_Friedland-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d228779-Reviews-Hotel_Marignan_Champs_Elysees-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d228835-Reviews-Hotel_Madeleine_Plaza-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d12829774-Reviews-Hotel_la_Nouvelle_Republique-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d197557-Reviews-Ducs_de_Bourgogne_Hotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d11869269-Reviews-Maison_Albar_Hotels_Le_Pont_Neuf-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d234633-Reviews-Hotel_Residence_Foch-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d231053-Reviews-Hotel_Marais_Bastille-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d7182695-Reviews-Maison_Souquet-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d197946-Reviews-Hotel_Bradford_Elysees_Astotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d6678144-Reviews-The_Peninsula_Paris-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d197576-Reviews-Hotel_Galileo-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d10159593-Reviews-Hotel_La_Comtesse-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d197551-Reviews-Hotel_Augustin_Astotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d233766-Reviews-Kyriad_Hotel_Paris_Bercy_Village-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d219994-Reviews-Hotel_de_Lutece-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d282185-Reviews-Hotel_France_Albion-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d10350792-Reviews-Hotel_Square_Louvois-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d188975-Reviews-Four_Seasons_Hotel_George_V-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d197528-Reviews-Le_Royal_Monceau_Raffles_Paris-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d10328342-Reviews-Hotel_Monge-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d266280-Reviews-Cler_Hotel-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d233725-Reviews-Hotel_Brighton_Esprit_de_France-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d188729-Reviews-Le_Bristol_Paris-Paris_Ile_de_France.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187147-d194284-Reviews-Hotel_Sainte_Beuve-Paris_Ile_de_France.html']
    
    links_3 = ['https://www.tripadvisor.com/Hotel_Review-g293916-d13134217-Reviews-Akara_Hotel-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d307155-Reviews-Shangri_La_Hotel_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d301334-Reviews-Pathumwan_Princess_Hotel-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297930-d12725009-Reviews-The_Marina_Phuket_Hotel-Patong_Kathu_Phuket.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d586655-Reviews-Millennium_Hilton_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d305496-Reviews-Rembrandt_Hotel_Suites_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d2318761-Reviews-Eastin_Grand_Hotel_Sathorn-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d300427-Reviews-Amari_Don_Muang_Airport_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d8822150-Reviews-Avani_Riverside_Bangkok_Hotel-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d308907-Reviews-Emporium_Suites_by_Chatrium-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d12245879-Reviews-The_Salil_Hotel_Sukhumvit_57_Thonglor-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d301613-Reviews-Chatrium_Residence_Sathon_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d10623691-Reviews-Compass_Skyview_Hotel_by_Compass_Hospitality-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293917-d7380321-Reviews-BED_Phrasingh_Adults_Only-Chiang_Mai.html',\
                'https://www.tripadvisor.com/Hotel_Review-g10804710-d12629455-Reviews-Avista_Grande_Phuket_Karon_MGallery_Hotel_Collection-Karon_Beach_Karon_Phuket.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d301808-Reviews-Royal_Orchid_Sheraton_Hotel_Towers-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d2049177-Reviews-Park_Plaza_Bangkok_Soi_18-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d1175010-Reviews-Lancaster_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d13341534-Reviews-Bangkok_Marriott_Hotel_The_Surawongse-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d1393554-Reviews-Sofitel_Bangkok_Sukhumvit-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g14902669-d6802617-Reviews-The_Yama_Hotel_Phuket-Ban_Kata_Karon_Phuket.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297930-d8275148-Reviews-Ramada_by_Wyndham_Phuket_Deevana_Patong-Patong_Kathu_Phuket.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1215780-d5985109-Reviews-Chanalai_Hillside_Resort_Karon_Beach_Phuket-Karon_Phuket.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d3169141-Reviews-Riva_Surya_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d302769-Reviews-The_Peninsula_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d308992-Reviews-Novotel_Bangkok_Sukhumvit_20-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d299546-Reviews-Novotel_Bangkok_on_Siam_Square-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297930-d3183240-Reviews-Avista_Hideaway_Phuket_Patong_MGallery_Hotel_Collection-Patong_Kathu_Phuket.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d308774-Reviews-Narai_Hotel-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d308724-Reviews-Ambassador_Hotel_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d302456-Reviews-The_Athenee_Hotel_A_Luxury_Collection_Hotel_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293917-d1147512-Reviews-Le_Meridien_Chiang_Mai-Chiang_Mai.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d7910036-Reviews-Movenpick_Hotel_Sukhumvit_15_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d608302-Reviews-Shanghai_Mansion_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d12918754-Reviews-Hyatt_Place_Bangkok_Sukhumvit-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293917-d15270748-Reviews-Phra_Singh_Village-Chiang_Mai.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d1026220-Reviews-Le_Meridien_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d310475-Reviews-Ibis_Styles_Bangkok_Khaosan_Viengtai-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d2421117-Reviews-SO_BANGKOK-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d301884-Reviews-Anantara_Siam_Bangkok_Hotel-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d2231933-Reviews-Viva_Garden_Serviced_Residence-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297914-d6514691-Reviews-The_Haven_Khao_Lak-Khao_Lak_Phang_Nga_Province.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d3793253-Reviews-Centara_Watergate_Pavillion_Hotel_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d12178556-Reviews-Park_Hyatt_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d1012383-Reviews-FuramaXclusive_Asoke-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d305223-Reviews-Grand_Hyatt_Erawan_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d2422237-Reviews-Grande_Centre_Point_Terminal_21-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d588954-Reviews-Bandara_Suites_Silom_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d1784761-Reviews-Maitria_Hotel_Sukhumvit_18_Bangkok_A_Chatrium_Collection-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3366878-d14782317-Reviews-U_Jomtien_Pattaya-Jomtien_Beach_Pattaya_Chonburi_Province.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d15019979-Reviews-Hotel_Nikko_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d850401-Reviews-Centara_Grand_at_CentralWorld-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d1724201-Reviews-Siam_Kempinski_Hotel_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293917-d301341-Reviews-Dusit_Princess_Chiang_Mai-Chiang_Mai.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293919-d6390812-Reviews-Amari_Pattaya-Pattaya_Chonburi_Province.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d14192482-Reviews-Hyatt_Regency_Bangkok_Sukhumvit-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d8404422-Reviews-Vince_Hotel-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1507054-d15152221-Reviews-Sea_Seeker_Krabi_Resort-Ao_Nang_Krabi_Town_Krabi_Province.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d302959-Reviews-Banyan_Tree_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d2624290-Reviews-The_Okura_Prestige_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d1784578-Reviews-Tower_Club_at_lebua-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d5988043-Reviews-Radisson_Blu_Plaza_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d305249-Reviews-Conrad_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d299585-Reviews-Centara_Grand_at_Central_Plaza_Ladprao_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293919-d1653641-Reviews-Pullman_Pattaya_Hotel_G-Pattaya_Chonburi_Province.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293916-d301411-Reviews-Mandarin_Oriental_Bangkok-Bangkok.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1507054-d15517242-Reviews-DusitD2_Ao_Nang_Krabi-Ao_Nang_Krabi_Town_Krabi_Province.html']
    
    links_4 = ['https://www.tripadvisor.com/Hotel_Review-g187791-d596437-Reviews-The_Inn_At_The_Roman_Forum-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d7976127-Reviews-A_Roma_Lifestyle_Hotel-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d2372788-Reviews-Dharma_Luxury_Hotel-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d1178206-Reviews-NH_Collection_Roma_Giustiniano-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d230612-Reviews-Hotel_Colosseum-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d208552-Reviews-Boutique_Hotel_Campo_de_Fiori-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d191099-Reviews-Albergo_del_Senato-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d203175-Reviews-Hotel_Barocco-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d232875-Reviews-The_Guardian-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d229054-Reviews-Hotel_Nazionale-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d231290-Reviews-Dharma_Style_Hotel-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d239263-Reviews-Hotel_Santa_Maria-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d275876-Reviews-Palazzo_Naiadi_The_Dedica_Anthology-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d316644-Reviews-Hotel_Raffaello-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d203094-Reviews-Best_Western_Hotel_President-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d236134-Reviews-Hotel_Abruzzi-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d228988-Reviews-Amalfi_Hotel-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d264648-Reviews-Hotel_delle_Province-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d543473-Reviews-Trilussa_Palace_Congress_Spa-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d205841-Reviews-Hotel_Alexandra-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d1084840-Reviews-QuodLibet-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d232851-Reviews-Hotel_De_Russie-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d1510471-Reviews-UNAHOTELS_Deco_Roma-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d203080-Reviews-The_Westin_Excelsior_Rome-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d650621-Reviews-Yes_Hotel-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d7134644-Reviews-Palazzo_Navona_Hotel-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d629750-Reviews-Hotel_Villa_Eur_Parco_dei_Pini-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d1729030-Reviews-IQ_Hotel_Roma-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d2213312-Reviews-The_First_Roma_Arte-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d557956-Reviews-Hotel_Golden-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d248861-Reviews-Hotel_Adriano-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d229043-Reviews-Hotel_Hiberia-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d2238199-Reviews-Hotel_Lunetta-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d577598-Reviews-Fabulous_Village-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d229087-Reviews-The_Inn_At_The_Spanish_Steps-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d4698947-Reviews-Appia_Antica_Resort-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d229991-Reviews-Best_Western_Plus_Hotel_Spring_House-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d208768-Reviews-Hotel_Italia-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d229102-Reviews-Hotel_delle_Nazioni-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d238031-Reviews-Hotel_Modigliani-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d3244512-Reviews-The_Independent_Hotel-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d203082-Reviews-Hotel_Regno-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d229036-Reviews-Hotel_Splendide_Royal-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d208553-Reviews-Kolbe_Hotel_Rome-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d190138-Reviews-Rome_Cavalieri_A_Waldorf_Astoria_Hotel-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d203087-Reviews-The_St_Regis_Rome-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d263860-Reviews-The_Beehive-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d3964772-Reviews-Rome_Times_Hotel-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d229422-Reviews-Crowne_Plaza_Rome_St_Peter_s-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d203093-Reviews-Best_Western_Plus_Hotel_Universo-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d581021-Reviews-Yellow_Square-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d228970-Reviews-FH_Grand_Hotel_Palatino-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d2163417-Reviews-Villa_Agrippina_Gran_Melia-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d230405-Reviews-NH_Collection_Roma_Vittorio_Veneto-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d203086-Reviews-Hotel_Morgana-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d228986-Reviews-Aldrovandi_Villa_Borghese-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d228975-Reviews-Hotel_Savoy_Roma-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d232888-Reviews-Ariston_Hotel-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d237193-Reviews-Hotel_Grifo-Rome_Lazio.html',\
                'https://www.tripadvisor.com/Hotel_Review-g187791-d231944-Reviews-Luxe_Rose_Garden_Hotel_Roma-Rome_Lazio.html']

    links_6 = ['https://www.tripadvisor.com/Hotel_Review-g186338-d1507196-Reviews-Apex_London_Wall_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d209229-Reviews-The_Bloomsbury-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193108-Reviews-Rosewood_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1845678-Reviews-Corinthia_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d187569-Reviews-The_Langham_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1026084-Reviews-The_Megaro_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d192031-Reviews-Novotel_London_Waterloo-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d192033-Reviews-Radisson_Blu_Edwardian_Hampshire_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d187685-Reviews-Claridge_s-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193113-Reviews-St_Giles_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d195279-Reviews-Corus_Hyde_Park_Hotel_Sure_Hotel_Collection_by_Best_Western-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d2539103-Reviews-Apex_Temple_Court_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d572859-Reviews-Apex_City_of_London_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d6822967-Reviews-The_Hoxton_Holborn-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d199849-Reviews-DoubleTree_by_Hilton_Hotel_London_Docklands_Riverside-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d8505156-Reviews-Hilton_London_Bankside-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193121-Reviews-The_Milestone_Hotel_and_Residences-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d309532-Reviews-Ace_Hotel_London_Shoreditch-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d4062677-Reviews-The_Resident_Soho-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d798951-Reviews-Andaz_London_Liverpool_Street-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193123-Reviews-Royal_Garden_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d3398027-Reviews-The_Westbridge_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d195185-Reviews-Hilton_London_Kensington-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d2554997-Reviews-The_Ampersand_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d195204-Reviews-DoubleTree_by_Hilton_Hotel_London_Victoria-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d291617-Reviews-Novotel_London_Excel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193133-Reviews-Holiday_Inn_London_Kensington_Forum-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d7179933-Reviews-Point_A_Hotel_London_Canary_Wharf-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d192030-Reviews-DoubleTree_by_Hilton_London_Angel_Kings_Cross_Formerly_Islington-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d192068-Reviews-Pullman_London_St_Pancras_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d278540-Reviews-Grand_Plaza_Serviced_Apartments-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d248802-Reviews-St_Ermin_s_Hotel_Autograph_Collection-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d3507299-Reviews-DoubleTree_by_Hilton_London_Greenwich-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d577546-Reviews-The_W14_Kensington-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d585836-Reviews-Hilton_London_Canary_Wharf-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d210361-Reviews-Premier_Inn_London_County_Hall_hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1600126-Reviews-Rafayel_on_the_Left_Bank-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d189041-Reviews-The_President_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d258944-Reviews-Park_Grand_London_Lancaster_Gate-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d6703394-Reviews-The_Z_Hotel_Piccadilly-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d289185-Reviews-Ibis_London_Excel_Docklands-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d187594-Reviews-The_Athenaeum_Hotel_Residences-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d236350-Reviews-Sofitel_London_St_James-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d188961-Reviews-Hotel_41-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193043-Reviews-Citadines_Trafalgar_Square_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d7193214-Reviews-Premier_Inn_London_City_Aldgate_hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d5823268-Reviews-Dorsett_Shepherds_Bush_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d255225-Reviews-Flemings_Mayfair-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d254575-Reviews-Luna_Simone_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d195285-Reviews-Sloane_Square_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193122-Reviews-The_Gore_Starhotels_Collezione-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d210066-Reviews-Alhambra_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d189735-Reviews-The_Resident_Kensington_formerly_The_Nadler_Kensington-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1719249-Reviews-H10_London_Waterloo-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d195198-Reviews-Crowne_Plaza_London_Kensington-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d9950629-Reviews-Hub_by_Premier_Inn_London_Spitalfields_Brick_Lane_hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1159072-Reviews-Novotel_London_Paddington_Hotel-London_England.html']

    links_7 = ['https://www.tripadvisor.com/Hotel_Review-g297701-d7022088-Reviews-The_Kayon_Resort_by_Pramana-Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d302653-Reviews-Melia_Bali-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297697-d7856221-Reviews-Amnaya_Resort_Kuta-Kuta_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297700-d10034774-Reviews-ARTOTEL_Sanur-Sanur_Denpasar_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d7368654-Reviews-The_Trans_Resort_Bali-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d5039960-Reviews-Sofitel_Bali_Nusa_Dua_Beach_Resort-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g608492-d301648-Reviews-Alila_Ubud-Payangan_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d301929-Reviews-Hilton_Bali_Resort-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d307570-Reviews-Grand_Hyatt_Bali-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d6635852-Reviews-Courtyard_by_Marriott_Bali_Seminyak_Resort-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d308410-Reviews-INAYA_Putri_Bali_Resort-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297696-d299128-Reviews-InterContinental_Bali_Resort-Jimbaran_South_Kuta_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g608487-d3528272-Reviews-The_Stones_Hotel_Legian_Bali_Autograph_Collection-Legian_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d1456498-Reviews-W_Bali_Seminyak-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297700-d307577-Reviews-Puri_Santrian-Sanur_Denpasar_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d1146351-Reviews-The_St_Regis_Bali_Resort-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297697-d1461739-Reviews-Kuta_Seaview_Boutique_Resort-Kuta_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d301649-Reviews-The_Legian_Bali-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d3633245-Reviews-Mulia_Resort-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d304528-Reviews-The_Westin_Resort_Nusa_Dua_Bali-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297696-d302249-Reviews-AYANA_Resort_and_Spa_Bali-Jimbaran_South_Kuta_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d3554169-Reviews-The_Mulia-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1465999-d308406-Reviews-Grand_Mirage_Resort_Thalasso_Spa_Bali-Tanjung_Benoa_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297701-d1168205-Reviews-Komaneka_at_Bisma-Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g608487-d1095744-Reviews-Bali_Mandira_Beach_Resort_Spa-Legian_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g608492-d7907965-Reviews-Padma_Resort_Ubud-Payangan_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297701-d309331-Reviews-Wapa_Di_Ume_Ubud-Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g2209545-d301883-Reviews-Four_Seasons_Resort_Bali_at_Sayan-Sayan_Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1465999-d320013-Reviews-Conrad_Bali-Tanjung_Benoa_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g608487-d1391615-Reviews-Pullman_Bali_Legian_Beach-Legian_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d1772090-Reviews-The_Seminyak_Beach_Resort_Spa-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297701-d2314151-Reviews-Komaneka_at_Rasa_Sayang-Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3932211-d307574-Reviews-Maya_Ubud_Resort_Spa-Peliatan_Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297701-d7376123-Reviews-Bisma_Eight-Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g608487-d7159969-Reviews-Mercure_Bali_Legian-Legian_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d450866-Reviews-Hotel_Vila_Lumbung-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1933359-d8293999-Reviews-Mandapa_a_Ritz_Carlton_Reserve-Kedewatan_Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g608487-d307610-Reviews-Padma_Resort_Legian-Legian_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297697-d307616-Reviews-Discovery_Kartika_Plaza_Hotel-Kuta_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1465999-d302638-Reviews-Sol_by_Melia_Benoa_Bali_All_Inclusive-Tanjung_Benoa_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1933359-d626311-Reviews-Komaneka_at_Tanggayuda-Kedewatan_Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297700-d325334-Reviews-Griya_Santrian-Sanur_Denpasar_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d590208-Reviews-Uma_Sapna-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g608487-d5287480-Reviews-Swiss_Belinn_Legian-Legian_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1933359-d550908-Reviews-The_Royal_Pita_Maha-Kedewatan_Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297700-d545823-Reviews-Prama_Sanur_Beach_Bali-Sanur_Denpasar_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d309355-Reviews-Impiana_Private_Villas_Seminyak-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d2038714-Reviews-Courtyard_Bali_Nusa_Dua_Resort-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d1456498-Reviews-W_Bali_Seminyak-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297697-d4587462-Reviews-Harper_Kuta-Kuta_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d302392-Reviews-The_Oberoi_Beach_Resort-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297701-d1174492-Reviews-Mason_Elephant_Lodge-Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297697-d588475-Reviews-Febri_s_Hotel_Spa-Kuta_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297701-d307569-Reviews-Komaneka_at_Monkey_Forest-Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g469404-d6556373-Reviews-Double_Six_Luxury_Hotel_Seminyak-Seminyak_Kuta_District_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1933359-d506292-Reviews-COMO_Uma_Ubud_Bali-Kedewatan_Ubud_Gianyar_Regency_Bali.html',\
                'https://www.tripadvisor.com/Hotel_Review-g297698-d6820858-Reviews-The_Ritz_Carlton_Bali-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html']

    links_10 = ['https://www.tripadvisor.com/Hotel_Review-g293924-d1592633-Reviews-Silk_Path_Hotel_Hanoi-Hanoi.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293924-d299548-Reviews-Sofitel_Legend_Metropole_Hanoi-Hanoi.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293924-d4042083-Reviews-JW_Marriott_Hotel_Hanoi-Hanoi.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293924-d1533551-Reviews-Sheraton_Hanoi_Hotel-Hanoi.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293924-d4609521-Reviews-Hanoi_Pearl_Hotel-Hanoi.html',\
                'https://www.tripadvisor.com/Hotel_Review-g293924-d1161184-Reviews-Movenpick_Hotel_Hanoi-Hanoi.html',\
                'https://www.tripadvisor.com/Hotel_Review-g666625-d256420-Reviews-Iberostar_Daiquiri-Cayo_Guillermo_Jardines_del_Rey_Archipelago_Ciego_de_Avila_Province_.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147275-d590422-Reviews-Blau_Varadero_Hotel-Varadero_Matanzas_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147275-d535717-Reviews-Iberostar_Selection_Varadero-Varadero_Matanzas_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147271-d148066-Reviews-Hotel_Nacional_de_Cuba-Havana_Ciudad_de_la_Habana_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g580450-d9603176-Reviews-Pullman_Cayo_Coco-Cayo_Coco_Jardines_del_Rey_Archipelago_Ciego_de_Avila_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147275-d14163595-Reviews-Melia_Internacional_Varadero_All_Inclusive-Varadero_Matanzas_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g1172156-d254186-Reviews-Memories_Jibacoa-Jibacoa_Mayabeque_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g737152-d263973-Reviews-Sol_Cayo_Largo-Cayo_Largo_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147271-d151444-Reviews-Iberostar_Parque_Central-Havana_Ciudad_de_la_Habana_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g670039-d12140756-Reviews-Sanctuary_at_Grand_Memories_Santa_Maria-Cayo_Santa_Maria_Villa_Clara_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g666625-d6220008-Reviews-Gran_Caribe_Club_Cayo_Guillermo-Cayo_Guillermo_Jardines_del_Rey_Archipelago_Ciego_de_A.html',\
                'https://www.tripadvisor.com/Hotel_Review-g666625-d256420-Reviews-Iberostar_Daiquiri-Cayo_Guillermo_Jardines_del_Rey_Archipelago_Ciego_de_Avila_Province_.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147275-d290690-Reviews-Be_Live_Experience_Las_Morlas-Varadero_Matanzas_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147271-d607380-Reviews-Saratoga_Hotel-Havana_Ciudad_de_la_Habana_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147275-d151553-Reviews-Sol_Palmeras-Varadero_Matanzas_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147275-d275185-Reviews-Starfish_Varadero-Varadero_Matanzas_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g2339909-d256417-Reviews-Gran_Club_Santa_Lucia-Playa_Santa_Lucia_Camaguey_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147271-d226417-Reviews-Melia_Cohiba-Havana_Ciudad_de_la_Habana_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147275-d151545-Reviews-Starfish_Cuatro_Palmas-Varadero_Matanzas_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g285731-d642754-Reviews-Iberostar_Heritage_Grand_Trinidad-Trinidad_Sancti_Spiritus_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g910832-d1211673-Reviews-Brisas_Guardalavaca_Hotel-Guardalavaca_Holguin_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147271-d6353319-Reviews-Hotel_NH_Capri_La_Habana-Havana_Ciudad_de_la_Habana_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147275-d151592-Reviews-Hotel_Roc_Arenas_Doradas-Varadero_Matanzas_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147271-d151450-Reviews-Hotel_Roc_Presidente-Havana_Ciudad_de_la_Habana_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147275-d151563-Reviews-Hotel_Roc_Barlovento-Varadero_Matanzas_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147271-d226421-Reviews-Hotel_Sevilla-Havana_Ciudad_de_la_Habana_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g580450-d263110-Reviews-Sol_Cayo_Coco-Cayo_Coco_Jardines_del_Rey_Archipelago_Ciego_de_Avila_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147275-d590422-Reviews-Blau_Varadero_Hotel-Varadero_Matanzas_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g580450-d266750-Reviews-TRYP_Cayo_Coco-Cayo_Coco_Jardines_del_Rey_Archipelago_Ciego_de_Avila_Province_Cuba.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3176298-d1076311-Reviews-Hard_Rock_Hotel_Casino_Punta_Cana-Bavaro_Punta_Cana_La_Altagracia_Province_Dominican_.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d16663802-Reviews-Hyatt_Ziva_Cap_Cana-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3176298-d259337-Reviews-Grand_Palladium_Bavaro_Suites_Resort_Spa-Bavaro_Punta_Cana_La_Altagracia_Province_Domi.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3176298-d1233228-Reviews-Iberostar_Grand_Bavaro-Bavaro_Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g663484-d253138-Reviews-Iberostar_Selection_Hacienda_Dominicus-Bayahibe_La_Altagracia_Province_Dominican_Republ.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3200043-d9762283-Reviews-Nickelodeon_Hotels_Resorts_Punta_Cana-Uvero_Alto_Punta_Cana_La_Altagracia_Province_Do.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d10175054-Reviews-Secrets_Cap_Cana_Resort_Spa-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d649099-Reviews-Zoetry_Agua_Punta_Cana-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3176298-d7307251-Reviews-The_Level_at_Melia_Caribe_Beach-Bavaro_Punta_Cana_La_Altagracia_Province_Dominican_Re.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d1717417-Reviews-The_Reserve_At_Paradisus_Palma_Real-Punta_Cana_La_Altagracia_Province_Dominican_Republ.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3200043-d8709413-Reviews-Excellence_El_Carmen-Uvero_Alto_Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147292-d149217-Reviews-Casa_de_Campo_Resort_Villas-La_Romana_La_Romana_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d218524-Reviews-Excellence_Punta_Cana-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d1199681-Reviews-Bahia_Principe_Luxury_Ambar-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d2399068-Reviews-Hotel_Riu_Palace_Bavaro-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d611703-Reviews-Catalonia_Royal_Bavaro-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g663484-d263551-Reviews-Catalonia_Gran_Dominicus-Bayahibe_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3176298-d9580294-Reviews-Majestic_Mirage_Punta_Cana-Bavaro_Punta_Cana_La_Altagracia_Province_Dominican_Republi.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3176298-d1068246-Reviews-Majestic_Elegance_Punta_Cana-Bavaro_Punta_Cana_La_Altagracia_Province_Dominican_Repub.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3176298-d1673192-Reviews-Barcelo_Bavaro_Palace-Bavaro_Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g16884615-d1022212-Reviews-Sanctuary_Cap_Cana-Cap_Cana_Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d283391-Reviews-Grand_Palladium_Palace_Resort_Spa_Casino-Punta_Cana_La_Altagracia_Province_Dominican_Re.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d4939796-Reviews-Royalton_Punta_Cana_Resort_Casino-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d583034-Reviews-Paradisus_Palma_Real_Golf_Spa_Resort-Punta_Cana_La_Altagracia_Province_Dominican_Republ.html',\
                'https://www.tripadvisor.com/Hotel_Review-g3176298-d313883-Reviews-Tropical_Princess_Beach_Resort_Spa-Bavaro_Punta_Cana_La_Altagracia_Province_Dominican_.html',\
                'https://www.tripadvisor.com/Hotel_Review-g147293-d10074376-Reviews-Now_Onyx_Punta_Cana-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',\
                'https://www.tripadvisor.com/Hotel_Review-g663484-d1163459-Reviews-Hilton_La_Romana_An_All_Inclusive_Adult_Resort-Bayahibe_La_Altagracia_Province_Dominic.html']

    # print(type(test))
    # test = ['https://www.tripadvisor.com/Hotel_Review-g60763-d7816364-Reviews-Executive_Hotel_LeSoleil-New_York_City_New_York.html',\
    #     'https://www.tripadvisor.com/Hotel_Review-g60763-d93437-Reviews-Hotel_Edison-New_York_City_New_York.html',\
    #     'https://www.tripadvisor.com/Hotel_Review-g60763-d3533197-Reviews-Hyatt_Union_Square_New_York-New_York_City_New_York.html']

    url_pgs = [links_10]
    pgs = ['links_10']

    for url_pg, pg in zip(url_pgs, pgs):
        print(pg)
        try:
            main(url_pg, pg)
        except:
            print("Something went wrong with " + pg)
            continue