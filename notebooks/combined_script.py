import pathlib
import pickle
import requests

import pandas as pd

DATA_DIR = pathlib.Path.cwd().parent / 'data'
print(DATA_DIR)

DATA_DIR.mkdir(parents=True, exist_ok=True)

raw_data_dir = DATA_DIR / 'raw'
raw_data_dir.mkdir(parents=True, exist_ok=True)
print(raw_data_dir)

raw_data_file_path = DATA_DIR / 'raw' / 'oxe.csv'
print(raw_data_file_path)

if not raw_data_file_path.exists():
    source_url = 'https://www.openintro.org/book/statdata/ames.csv'
    headers = {
        'User-Agent': \
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) ' \
            'AppleWebKit/537.36 (KHTML, like Gecko) ' \
            'Chrome/39.0.2171.95 Safari/537.36',
    }
    response = requests.get(source_url, headers=headers)
    csv_content = response.content.decode()
    with open(raw_data_file_path, 'w', encoding='utf8') as file:
        file.write(csv_content)

filesize = raw_data_file_path.stat().st_size
print(f'This file has {filesize} bytes')

raw_data = pd.read_csv(raw_data_file_path)

raw_data.shape

raw_data.head()

data = raw_data.copy()

data.info()

data.dtypes.value_counts()

ignore_variables = [
    'Order',
    'PID',
]

continuous_variables = [
    'Lot.Frontage',
    'Lot.Area',
    'Mas.Vnr.Area',
    'BsmtFin.SF.1',
    'BsmtFin.SF.2',
    'Bsmt.Unf.SF',
    'Total.Bsmt.SF',
    'X1st.Flr.SF',
    'X2nd.Flr.SF',
    'Low.Qual.Fin.SF',
    'Gr.Liv.Area',
    'Garage.Area',
    'Wood.Deck.SF',
    'Open.Porch.SF',
    'Enclosed.Porch',
    'X3Ssn.Porch',
    'Screen.Porch',
    'Pool.Area',
    'Misc.Val',
    'SalePrice',
]

discrete_variables = [
    'Year.Built',
    'Year.Remod.Add',
    'Bsmt.Full.Bath',
    'Bsmt.Half.Bath',
    'Full.Bath',
    'Half.Bath',
    'Bedroom.AbvGr',
    'Kitchen.AbvGr',
    'TotRms.AbvGrd',
    'Fireplaces',
    'Garage.Yr.Blt',
    'Garage.Cars',
    'Mo.Sold',
    'Yr.Sold',
]

ordinal_variables = [
    'Lot.Shape',
    'Utilities',
    'Land.Slope',
    'Overall.Qual',
    'Overall.Cond',
    'Exter.Qual',
    'Exter.Cond',
    'Bsmt.Qual',
    'Bsmt.Cond',
    'Bsmt.Exposure',
    'BsmtFin.Type.1',
    'BsmtFin.Type.2',
    'Heating.QC',
    'Electrical',
    'Kitchen.Qual',
    'Functional',
    'Fireplace.Qu',
    'Garage.Finish',
    'Garage.Qual',
    'Garage.Cond',
    'Paved.Drive',
    'Pool.QC',
    'Fence',
]

categorical_variables = [
    'MS.SubClass',
    'MS.Zoning',
    'Street',
    'Alley',
    'Land.Contour',
    'Lot.Config',
    'Neighborhood',
    'Condition.1',
    'Condition.2',
    'Bldg.Type',
    'House.Style',
    'Roof.Style',
    'Roof.Matl',
    'Exterior.1st',
    'Exterior.2nd',
    'Mas.Vnr.Type',
    'Foundation',
    'Heating',
    'Central.Air',
    'Garage.Type',
    'Misc.Feature',
    'Sale.Type',
    'Sale.Condition',
]

data.drop(columns=['Order', 'PID'], inplace=True)

for col in continuous_variables:
    data[col] = data[col].astype('float64')

for col in categorical_variables:
    data[col] = data[col].astype('category')

for col in discrete_variables:
    data[col] = data[col].astype('float64')

data[ordinal_variables].info()

category_orderings = {
    'Lot.Shape': [
        'Reg',
        'IR1',
        'IR2',
        'IR3',
    ],
    'Utilities': [
        'AllPub',
        'NoSewr',
        'NoSeWa',
        'ELO',
    ],
    'Land.Slope': [
        'Gtl',
        'Mod',
        'Sev',
    ],
    'Overall.Qual': [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ],
    'Overall.Cond': [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ],
    'Exter.Qual': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Exter.Cond': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Bsmt.Qual': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Bsmt.Cond': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Bsmt.Exposure': [
        'Gd',
        'Av',
        'Mn',
        'No',
        'NA',
    ],
    'BsmtFin.Type.1': [
        'GLQ',
        'ALQ',
        'BLQ',
        'Rec',
        'LwQ',
        'Unf',
    ],
    'BsmtFin.Type.2': [
        'GLQ',
        'ALQ',
        'BLQ',
        'Rec',
        'LwQ',
        'Unf',
    ],
    'Heating.QC': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Electrical': [
        'SBrkr',
        'FuseA',
        'FuseF',
        'FuseP',
        'Mix',
    ],
    'Kitchen.Qual': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Functional': [
        'Typ',
        'Min1',
        'Min2',
        'Mod',
        'Maj1',
        'Maj2',
        'Sev',
        'Sal',
    ],
    'Fireplace.Qu': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Garage.Finish': [
        'Fin',
        'RFn',
        'Unf',
    ],
    'Garage.Qual': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Garage.Cond': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',    
    ],
    'Paved.Drive': [
        'Y',
        'P',
        'N',
    ],
    'Pool.QC': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
    ],
    'Fence': [
        'GdPrv',
        'MnPrv',
        'GdWo',
        'MnWw',
    ],
}


for col, orderings in category_orderings.items():
    data[col] = data[col] \
        .astype('category') \
        .cat \
        .set_categories(orderings, ordered=True)

data \
    .select_dtypes('category') \
    .describe() \
    .transpose() \
    .sort_values(by='count', ascending=True)

data \
    .select_dtypes('category') \
    .describe() \
    .transpose() \
    .sort_values(by='unique', ascending=True)

data \
    .select_dtypes('number') \
    .describe() \
    .transpose() \
    .sort_values(by='count', ascending=True)

processed_dir = DATA_DIR / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

processed_file_path = processed_dir / 'ames_with_correct_types2.pkl'

with open(processed_file_path, 'wb') as file:
    pickle.dump(
        [
            data,
            continuous_variables,
            discrete_variables,
            ordinal_variables,
            categorical_variables,
        ],
        file,
    )


import pathlib
import pickle


import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', 500)

DATA_DIR = pathlib.Path.cwd().parent / 'data'
print(DATA_DIR)

processed_file_path = DATA_DIR / 'processed' / 'ames_with_correct_types2.pkl'

with open(processed_file_path, 'rb') as file:
    (
        data,
        continuous_variables,
        discrete_variables,
        ordinal_variables,
        categorical_variables,
    ) = pickle.load(file)


def plot_categoricals(data, cols, sorted=True):
    summary = data[cols] \
        .describe() \
        .transpose() \
        .sort_values(by='count')

    print(summary)



data['MS.Zoning'].unique()

data['MS.Zoning'].value_counts()

selection = ~(data['MS.Zoning'].isin(['A (agr)', 'C (all)', 'I (all)']))
selection.value_counts()

data = data[selection]

data['MS.Zoning'] = data['MS.Zoning'].cat.remove_unused_categories()

data['MS.Zoning'].value_counts()

data['Sale.Type'].value_counts()

data['Sale.Type'].unique()

data['Sale.Condition'].value_counts()

processed_data = data.copy()

def remap_categories(
    series: pd.Series,
    old_categories: tuple[str],
    new_category: str,
) -> pd.Series:
    # Add the new category to the list of valid categories.
    series = series.cat.add_categories(new_category)

    # Set all items of the old categories as the new category.
    remapped_items = series.isin(old_categories)
    series.loc[remapped_items] = new_category

    # Clean up the list of categories, the old categories no longer exist.
    series = series.cat.remove_unused_categories()

    return series

processed_data['Sale.Type'] = remap_categories(
    series=processed_data['Sale.Type'],
    old_categories=('WD ', 'CWD', 'VWD'),
    new_category='GroupedWD',
)

processed_data['Sale.Type'] = remap_categories(
    series=processed_data['Sale.Type'],
    old_categories=('COD', 'ConLI', 'Con', 'ConLD', 'Oth', 'ConLw'),
    new_category='Other',
)

processed_data['Sale.Type'].value_counts()

data = processed_data

data['Street'].value_counts()

data = data.drop(columns='Street')

data['Condition.1'].value_counts()

data['Condition.2'].value_counts()

pd.crosstab(data['Condition.1'], data['Condition.2'])

processed_data = data.copy()

for col in ('Condition.1', 'Condition.2'):
    processed_data[col] = remap_categories(
        series=processed_data[col],
        old_categories=('RRAn', 'RRAe', 'RRNn', 'RRNe'),
        new_category='Railroad',
    )
    processed_data[col] = remap_categories(
        series=processed_data[col],
        old_categories=('Feedr', 'Artery'),
        new_category='Roads',
    )
    processed_data[col] = remap_categories(
        series=processed_data[col],
        old_categories=('PosA', 'PosN'),
        new_category='Positive',
    )

processed_data['Condition.1'].value_counts()

processed_data['Condition.2'].value_counts()

pd.crosstab(processed_data['Condition.1'], processed_data['Condition.2'])

processed_data['Condition'] = pd.Series(
    index=processed_data.index,
    dtype=pd.CategoricalDtype(categories=(
        'Norm',
        'Railroad',
        'Roads',
        'Positive',
        'RoadsAndRailroad',
    )),
)

norm_items = processed_data['Condition.1'] == 'Norm'
processed_data['Condition'][norm_items] = 'Norm'

railroad_items = \
    (processed_data['Condition.1'] == 'Railroad') \
    & (processed_data['Condition.2'] == 'Norm')
processed_data['Condition'][railroad_items] = 'Railroad'

roads_items = \
    (processed_data['Condition.1'] == 'Roads') \
    & (processed_data['Condition.2'] != 'Railroad')
processed_data['Condition'][roads_items] = 'Roads'

positive_items = processed_data['Condition.1'] == 'Positive'
processed_data['Condition'][positive_items] = 'Positive'

roads_and_railroad_items = \
    ( \
        (processed_data['Condition.1'] == 'Railroad') \
        & (processed_data['Condition.2'] == 'Roads')
    ) \
    | ( \
        (processed_data['Condition.1'] == 'Roads') \
        & (processed_data['Condition.2'] == 'Railroad') \
    )
processed_data['Condition'][roads_and_railroad_items] = 'RoadsAndRailroad'

processed_data['Condition'].value_counts()

processed_data = processed_data.drop(columns=['Condition.1', 'Condition.2'])

data = processed_data


data['HasShed'] = data['Misc.Feature'] == 'Shed'
data = data.drop(columns='Misc.Feature')

data['HasShed'].value_counts()

data['HasAlley'] = ~data['Alley'].isna()
data = data.drop(columns='Alley')

data['HasAlley'].value_counts()


data['Exterior.2nd'] = remap_categories(
    series=data['Exterior.2nd'],
    old_categories=('Brk Cmn', ),
    new_category='BrkComm',
)
data['Exterior.2nd'] = remap_categories(
    series=data['Exterior.2nd'],
    old_categories=('CmentBd', ),
    new_category='CemntBd',
)
data['Exterior.2nd'] = remap_categories(
    series=data['Exterior.2nd'],
    old_categories=('Wd Shng', ),
    new_category='WdShing',
)


for col in ('Exterior.1st', 'Exterior.2nd'):
    categories = data[col].cat.categories
    data[col] = data[col].cat.reorder_categories(sorted(categories))

pd.crosstab(data['Exterior.1st'], data['Exterior.2nd'])

processed_data = data.copy()

mat_count = processed_data['Exterior.1st'].value_counts()
mat_count

rare_materials = list(mat_count[mat_count < 40].index)
rare_materials

processed_data['Exterior'] = remap_categories(
    series=processed_data['Exterior.1st'],
    old_categories=rare_materials,
    new_category='Other',
)
processed_data = processed_data.drop(columns=['Exterior.1st', 'Exterior.2nd'])

processed_data['Exterior'].value_counts()

data = processed_data


data = data.drop(columns='Heating')


data = data.drop(columns='Roof.Matl')

data['Roof.Style'] = remap_categories(
    series=data['Roof.Style'],
    old_categories=[
        'Flat',
        'Gambrel',
        'Mansard',
        'Shed',
    ],
    new_category='Other',
)

data['Roof.Style'].value_counts()

data['Mas.Vnr.Type'].info()

data['Mas.Vnr.Type'].value_counts()

data['Mas.Vnr.Type'] = remap_categories(
    series=data['Mas.Vnr.Type'],
    old_categories=[
        'BrkCmn',
        'CBlock',
    ],
    new_category='Other',
)

data['Mas.Vnr.Type'] = data['Mas.Vnr.Type'].cat.add_categories('None')
data['Mas.Vnr.Type'][data['Mas.Vnr.Type'].isna()] = 'None'


data['Mas.Vnr.Type'].value_counts()


data['MS.SubClass'] = remap_categories(
    series=data['MS.SubClass'],
    old_categories=[75, 45, 180, 40, 150],
    new_category='Other',
)

data['MS.SubClass'].value_counts()


data['Foundation'] = remap_categories(
    series=data['Foundation'],
    old_categories=['Slab', 'Stone', 'Wood'],
    new_category='Other',
)

data['Neighborhood'].value_counts()

selection = ~data['Neighborhood'].isin([
    'Blueste',
    'Greens',
    'GrnHill',
    'Landmrk',
])
data = data[selection]

data['Neighborhood'] = data['Neighborhood'].cat.remove_unused_categories()

data['Neighborhood'].value_counts()

data['Garage.Type'].info()

data['Garage.Type'].value_counts()

data['Garage.Type'] = data['Garage.Type'].cat.add_categories(['NoGarage'])
data['Garage.Type'][data['Garage.Type'].isna()] = 'NoGarage'

data['Garage.Type'].value_counts()

all_categorical = data.select_dtypes('category').columns

new_categorical_variables = [ \
    col for col in all_categorical \
    if not col in ordinal_variables \
]



data = data.drop(columns='Utilities')

data = data.drop(columns='Pool.QC')

data['Fence'].value_counts().sort_index()

old_categories = list(data['Fence'].cat.categories)
old_categories

new_categories = old_categories + ['NoFence']
new_categories

data['Fence'] = data['Fence'].cat.set_categories(new_categories)

data['Fence'].dtype

data['Fence'][data['Fence'].isna()] = 'NoFence'

data['Fence'].value_counts().sort_index()

data['Fireplace.Qu'].value_counts().sort_index()

data['Fireplaces'].value_counts()

data = data.drop(columns='Fireplace.Qu')



# data = data.drop(columns=['Garage.Cond', 'Garage.Qual'])

data['Garage.Finish'] = data['Garage.Finish'] \
    .cat \
    .as_unordered() \
    .cat \
    .add_categories(['NoGarage'])
data['Garage.Finish'][data['Garage.Finish'].isna()] = 'NoGarage'

data['Garage.Finish'].value_counts()

data['Garage.Finish'].dtype

data['Garage.Finish'].cat.ordered

data['Electrical'].isna().value_counts()


data['Electrical'][data['Electrical'].isna()] = 'SBrkr'

ordinal_columns = [col for col in data.select_dtypes('category') if data[col].cat.ordered]

data[ordinal_columns].info()


data['Bsmt.Exposure'].unique()

data['Bsmt.Exposure'][data['Bsmt.Exposure'].isna()] = 'NA'
data['Bsmt.Exposure'] = data['Bsmt.Exposure'] \
    .cat \
    .as_unordered() \
    .cat \
    .remove_unused_categories()

for col in ('Bsmt.Qual', 'Bsmt.Cond', 'BsmtFin.Type.1', 'BsmtFin.Type.2'):
    data[col] = data[col].cat.add_categories(['NA'])
    data[col][data[col].isna()] = 'NA'
    data[col] = data[col] \
        .cat \
        .as_unordered() \
        .cat \
        .remove_unused_categories()


data['Bsmt.Cond'][data['Bsmt.Cond'] == 'Po'] = 'Fa'
data['Bsmt.Cond'][data['Bsmt.Cond'] == 'Ex'] = 'Gd'
data['Bsmt.Cond'] = data['Bsmt.Cond'].cat.remove_unused_categories()

data['Bsmt.Cond'].value_counts()

data[ordinal_columns].info()

data['SalePrice'].describe()

data['SalePrice'] = data['SalePrice'].apply(np.log10)

data['SalePrice'].describe()

data['Lot.Frontage'].info()

missing_lot_frontage = data['Lot.Frontage'].isna()

data['MS.SubClass'][missing_lot_frontage].value_counts()

data['Lot.Config'][missing_lot_frontage].value_counts()

data['Land.Contour'][missing_lot_frontage].value_counts()


data[['Lot.Frontage', 'Lot.Area']].corr()

aux_data = data[['Lot.Frontage', 'Lot.Area']].copy()
aux_data['Sqrt.Lot.Area'] = aux_data['Lot.Area'].apply(np.sqrt)

aux_data[['Lot.Frontage', 'Sqrt.Lot.Area']].corr()



data['Lot.Frontage'] = data['Lot.Frontage'].fillna(data['Lot.Frontage'].median())

data['Lot.Frontage'].info()



data['Garage.Yr.Blt'].describe()

garage_age = data['Yr.Sold'] - data['Garage.Yr.Blt']
garage_age.describe()

data[garage_age < 0.0].transpose()

garage_age[garage_age < 0.0] = 0.0

data = data.drop(columns='Garage.Yr.Blt')
data['Garage.Age'] = garage_age

data['Garage.Age'].info()

data['Garage.Type'][data['Garage.Age'].isna()].value_counts()

data['Garage.Age'] = data['Garage.Age'].fillna(data['Garage.Age'].median())

data[['Year.Remod.Add', 'Year.Built', 'Yr.Sold']].describe()

remod_age = data['Yr.Sold'] - data['Year.Remod.Add']
remod_age.describe()

data[remod_age < 0.0].transpose()

remod_age[remod_age < 0.0] = 0.0

house_age = data['Yr.Sold'] - data['Year.Built']
house_age.describe()

data[house_age < 0.0].transpose()

house_age[house_age < 0.0] = 0.0

data = data.drop(columns=['Year.Remod.Add'])
data['Remod.Age'] = remod_age
data['House.Age'] = house_age

data['Mas.Vnr.Area'].info()

data['Mas.Vnr.Type'][data['Mas.Vnr.Area'].isna()].value_counts()

data.loc[data['Mas.Vnr.Area'].isna(), 'Mas.Vnr.Area'] = 0.0

num_houses = data.shape[0]
num_houses_with_pool = data[data['Pool.Area'] > 0].shape[0]
print(f'Out of {num_houses} houses, only {num_houses_with_pool} have a pool.')

data.info()

data = data.dropna(axis=0)

data.info()

for col in data.select_dtypes('category').columns:
    data[col] = data[col].cat.remove_unused_categories()

numerical_data = data.select_dtypes('number').drop(columns='SalePrice').copy()
target = data['SalePrice'].copy()

numerical_data.corrwith(target).sort_values()


categorical_columns = data.select_dtypes('category').columns


corr = data.corr(numeric_only=True)
corr

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

# import numpy as np

# # 1. Criação de Variáveis Combinadas
# # Área Total da Casa
# data['TotalSF'] = data['X1st.Flr.SF'] + data['X2nd.Flr.SF'] + data['Total.Bsmt.SF']

# # Total de Banheiros
# data['TotalBath'] = data['Full.Bath'] + 0.5 * data['Half.Bath'] + data['Bsmt.Full.Bath'] + 0.5 * data['Bsmt.Half.Bath']

# # 2. Transformações Matemáticas
# # Log da Área Total

# data['Garage.CarsPerArea'] = data['Garage.Cars'] / data['Garage.Area']
# data['Garage.CarsPerArea'] = data['Garage.CarsPerArea'].fillna(value=0)

# columns_to_standardize = ['Lot.Area', 'X1st.Flr.SF', 'Bsmt.Unf.SF', 'Total.Bsmt.SF', 'Lot.Frontage']
# columns_to_drop = ['X2nd.Flr.SF', 'X3Ssn.Porch', 'BsmtFin.SF.2', 'Bsmt.Half.Bath','Mas.Vnr.Area']

# data = data.drop(columns=columns_to_drop)

# #standalize columns
# for column in columns_to_standardize:
#     data[column] = (data[column] - data[column].mean()) / data[column].std()

# # 4. Interação entre Variáveis

# # Interação entre Qualidade e Condição


# # Mostrando as primeiras linhas do dataframe modificado
# data.head()

clean_data_path = DATA_DIR / 'processed' / 'ames_clean2.pkl'

with open(clean_data_path, 'wb') as file:
    pickle.dump(data, file)




import pickle
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = pathlib.Path.cwd().parent / 'data'
print(DATA_DIR)

clean_data_path = DATA_DIR / 'processed' / 'ames_clean2.pkl'

with open(clean_data_path, 'rb') as file:
    data = pickle.load(file)

data = data.copy()

data['TotalSF'] = data['X1st.Flr.SF'] + data['X2nd.Flr.SF'] + data['Total.Bsmt.SF']

data['TotalBath'] = data['Full.Bath'] + 0.5 * data['Half.Bath'] + data['Bsmt.Full.Bath'] + 0.5 * data['Bsmt.Half.Bath']

data["All.Porch.SF"] = data["Open.Porch.SF"] + data["Enclosed.Porch"] + data["X3Ssn.Porch"] + data["Screen.Porch"]

data['TotalSFLog'] = np.log(data['TotalSF'])


data["All.SF"] = data["Gr.Liv.Area"] + data["Total.Bsmt.SF"]


data["AllSF-s2"] = data["All.SF"] ** 2
# data["Overall.Qual-s2"] = data["Overall.Qual"] ** 2
# data["All.FlrsSF-Sq"] = np.sqrt(data["AllFlrsSF"])
data["GrLivArea-s2"] = data["Gr.Liv.Area"] ** 2




(data['Gr.Liv.Area'] < 4000).value_counts()

data = data[data['Gr.Liv.Area'] < 4000]

corr = data.corr(numeric_only=True)
sns.heatmap(corr[corr>0.8], xticklabels=corr.columns, yticklabels=corr.columns)

#generete model_data csv
model_data_path = DATA_DIR / 'processed' / 'ames_model_data.csv'

data.to_csv(model_data_path, index=False)
