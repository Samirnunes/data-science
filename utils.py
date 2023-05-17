import pandas as pd

def import_air_pollution_data(place_abbreviation: str, years: list):
    '''
    - place_abbreviation: string that determines the place from which data is from. 
    It's extracted from the file's name. Ex: SP201501.csv --> place_abbreviation = 'SP'.

    - years: list of strings of the years from which we want to get data. 
    Restricted to the years that exists in the names of the data files. 
    Ex: SP201501.csv, SP201502.csv, SP201601.csv, SP201602.csv --> years = ['2015', '2016'].

    returns: pandas dataframe with all the data from the specified place and years.
    '''

    sp_pol = {}

    for y in years:
        first_df = pd.read_csv(f'data/{place_abbreviation}{y}01.csv', encoding = 'latin-1')
        sec_df = pd.read_csv(f'data/{place_abbreviation}{y}02.csv', encoding = 'latin-1')
        sp_pol[y] = pd.concat([first_df, sec_df])

    # Agora uniremos todos os dados em um mesmo dataframe que, por simplicidade, chamaremos de data.
    data = sp_pol[years[0]]
        
    for y in years[1:]:
        data = pd.concat([data, sp_pol[y]])

    data.reset_index(drop = True, inplace = True)
    data['ID'] = list(data.index)
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]

    print('Dados importados e reunidos no dataframe data.')

    return data
