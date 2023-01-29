# selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException as SEException, NoSuchElementException as NSEException
from webdriver_manager.chrome import ChromeDriverManager

import csv
import time
import re # regular expressions

# initialize webdriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
driver = webdriver.Chrome(ChromeDriverManager().install())

driver.get("https://www.nba.com/stats/players/traditional?PerMode=PerGame&dir=A&sort=NBA_FANTASY_PTS") # load website

# format ouput file
players_file = csv.writer(open('player_stats.csv', 'w', newline=''))
players_file.writerow(['AGE', 'W', 'L', 'MIN', 'PTS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'FP', '+/-'])

for season in range(27): # iterate accross all seasons in data
   # This should have worked, but didn't. May revisit in the future.
   # WebDriverWait(driver, 15, ignored_exceptions=(SEException, NSEException)).until(EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div[3]/section[1]/div/div/div[1]/label/div/select')))

   time.sleep(1.5) # wait for elements to load (not very elegant I know)
   # select the correct season
   season_list = driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div[3]/section[1]/div/div/div[1]/label/div/select')
   season_selector = Select(season_list)
   season_selector.select_by_index(season)


   # WebDriverWait(driver, 15, ignored_exceptions=(SEException, NSEException)).until(EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select')))

   time.sleep(1.5)
   # select "all" from page count. this shows all of the data on one page
   drop_down_list = driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select')
   page_selector = Select(drop_down_list)
   page_selector.select_by_value('-1')
   
   # WebDriverWait(driver, 15, ignored_exceptions=(SEException, NSEException)).until(EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table/tbody')))

   time.sleep(1.5)
   # grab the table of players
   players = driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table/tbody').text.split('\n')
   # for each player, reformat the raw data into a 2D array
   for player in players:

      temp = player.split(' ')
      digits = re.compile('[0-9]')
      # since we split by spaces, we have to account for variable-length names
      if digits.match(temp[3][0]) != None: # accounts for players with one name (Nene)
         player = [0] * 31
         player[:3] = temp[:3]
         player[3] = ''
         player[4:] = temp[3:]
      elif digits.match(temp[5][0]) == None: # accounts for players with 4 names (Luc Mbah a Moute)
         player = [0] * 31
         player[:4] = temp[:4]
         player[4:] = temp[6:]
      elif digits.match(temp[4][0]) == None: # accounts for players with 3 names
         player = [0] * 31
         player[:4] = temp[:4]
         player[4:] = temp[5:]
      else:
         player = temp

      # store desired stats in an array
      stats = [ 
         player[4], # age
         player[6], # wins
         player[7], # losses
         player[8], # minutes played
         player[9], # points per game
         player[10], # field goals made
         player[11], # field goals attempted
         player[13], # 3 pointers made
         player[14], # 3 pointers attempted
         player[16], # freethrows made
         player[17], # freethrows attempted
         player[19], # offensive rebounds
         player[20], # defensive rebounds
         player[22], # assists
         player[23], # turnovers
         player[24], # steals
         player[25], # blocks
         player[26], # personal fouls
         player[27], # fantasy points
         player[30] # plus-minus
      ]
      players_file.writerow(stats) # write to our output csv

driver.quit()
