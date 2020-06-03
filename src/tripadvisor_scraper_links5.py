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

    links_5 = ['https://www.tripadvisor.com/Hotel_Review-g186338-d8147345-Reviews-InterContinental_London_The_O2-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1657415-Reviews-Park_Plaza_Westminster_Bridge_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193143-Reviews-Park_Grand_London_Kensington-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193112-Reviews-Strand_Palace-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d192064-Reviews-The_Chesterfield_Mayfair-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193104-Reviews-Amba_Hotel_Charing_Cross-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193097-Reviews-The_Tower_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193062-Reviews-Park_Grand_Paddington_Court-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d10810215-Reviews-Montcalm_Royal_London_House_City_of_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d265539-Reviews-St_James_Court_A_Taj_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193079-Reviews-The_Royal_Horseguards-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193129-Reviews-Danubius_Hotel_Regents_Park-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d10340857-Reviews-Park_Plaza_London_Waterloo-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1946018-Reviews-DoubleTree_by_Hilton_Hotel_London_Tower_of_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d195217-Reviews-K_K_Hotel_George-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d6484754-Reviews-Shangri_La_Hotel_At_The_Shard_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d4945399-Reviews-Qbic_Hotel_London_City-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d218403-Reviews-Grand_Royale_London_Hyde_Park-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d192069-Reviews-The_Montcalm_London_Marble_Arch-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d6161763-Reviews-Sea_Containers_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d187735-Reviews-The_May_Fair_A_Radisson_Collection_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193617-Reviews-The_Marylebone-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1924847-Reviews-Studios2Let_Serviced_Apartments_Cartwright_Gardens-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d11621939-Reviews-Novotel_London_Canary_Wharf-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d192036-Reviews-The_Montague_on_The_Gardens-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d3240599-Reviews-Point_A_Hotel_London_Kings_Cross_St_Pancras-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d7182667-Reviews-M_by_Montcalm_Shoreditch_London_Tech_City-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193084-Reviews-Amba_Hotel_Marble_Arch-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d243430-Reviews-London_House_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d651411-Reviews-Radisson_Blu_Edwardian_New_Providence_Wharf_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d187591-Reviews-The_Ritz_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d3523344-Reviews-Ibis_London_Blackfriars_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1477336-Reviews-Club_Quarters_Hotel_Trafalgar_Square-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1371050-Reviews-The_Kensington-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d3226266-Reviews-Point_A_Hotel_London_Paddington-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193672-Reviews-Novotel_London_West-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d199868-Reviews-The_Rubens_at_the_Palace-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d672863-Reviews-Park_Plaza_County_Hall_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d6603498-Reviews-Sunborn_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193063-Reviews-Best_Western_Mornington_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d612985-Reviews-Palmers_Lodge_Swiss_Cottage-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d192034-Reviews-Radisson_Blu_Edwardian_Mercer_Street_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d195221-Reviews-Mercure_London_Paddington_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d2458180-Reviews-Point_A_Hotel_London_Liverpool_Street-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d614236-Reviews-Hilton_London_Tower_Bridge-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d621785-Reviews-The_Hoxton_Shoreditch-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d559155-Reviews-Taj_51_Buckingham_Gate_Suites_and_Residences-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d8625676-Reviews-The_Resident_Victoria_formerly_The_Nadler_Victoria-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d3523344-Reviews-Ibis_London_Blackfriars_Hotel-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d239656-Reviews-Club_Quarters_Hotel_St_Paul_s-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d5961318-Reviews-Hampton_by_Hilton_London_Waterloo-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d187686-Reviews-The_Savoy-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d2309633-Reviews-Conrad_London_St_James-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d3199601-Reviews-CitizenM_London_Bankside-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d3164384-Reviews-Park_Grand_London_Hyde_Park-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193045-Reviews-The_Waldorf_Hilton_London-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d1061544-Reviews-Travelodge_London_Central_Southwark-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d192139-Reviews-Ibis_London_Earls_Court-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d193125-Reviews-Millennium_Gloucester_Hotel_London_Kensington-London_England.html',\
                'https://www.tripadvisor.com/Hotel_Review-g186338-d2630895-Reviews-ME_London-London_England.html']

    # print(type(test))
    # test = ['https://www.tripadvisor.com/Hotel_Review-g60763-d7816364-Reviews-Executive_Hotel_LeSoleil-New_York_City_New_York.html',\
    #     'https://www.tripadvisor.com/Hotel_Review-g60763-d93437-Reviews-Hotel_Edison-New_York_City_New_York.html',\
    #     'https://www.tripadvisor.com/Hotel_Review-g60763-d3533197-Reviews-Hyatt_Union_Square_New_York-New_York_City_New_York.html']

    url_pgs = [links_5]
    pgs = ['links_5']

    for url_pg, pg in zip(url_pgs, pgs):
        print(pg)
        try:
            main(url_pg, pg)
        except:
            print("Something went wrong with " + pg)
            continue