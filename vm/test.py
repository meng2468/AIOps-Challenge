import csv
import datetime

date_pref = datetime.datetime.now().strftime('%d-%m:%H')
dics = {'sdd': 123, 'dgf': 324123, 'fd': 3333}

print(dics.values())
with open(date_pref+'_esb.csv', 'a') as csv_file:
    print(csv.reader(csv_file).line_num)
    csv.writer(csv_file).writerow([str(x) for x in dics.values()])

print([str(x) for x in dics.values()])