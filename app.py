from flask import Flask, request, jsonify
import joblib
import pandas as pd
import subprocess
import os

app = Flask(__name__)

# Caminho onde o arquivo de dados será salvo
DATA_PATH = 'data/raw/oxe.csv'


# Carregar o modelo
model = joblib.load('notebooks/ridge.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    colunas_finais = ['Lot.Frontage','Lot.Area','Lot.Shape','Land.Slope','Overall.Qual','Overall.Cond','Year.Built','Mas.Vnr.Area','Exter.Qual','Exter.Cond','BsmtFin.SF.1','BsmtFin.SF.2','Bsmt.Unf.SF','Total.Bsmt.SF','Heating.QC','Electrical','X1st.Flr.SF','X2nd.Flr.SF','Low.Qual.Fin.SF','Gr.Liv.Area','Bsmt.Full.Bath','Bsmt.Half.Bath','Full.Bath','Half.Bath','Bedroom.AbvGr','Kitchen.AbvGr','Kitchen.Qual','TotRms.AbvGrd','Functional' 'Fireplaces','Garage.Cars','Garage.Area','Garage.Qual','Garage.Cond','Paved.Drive','Wood.Deck.SF','Open.Porch.SF','Enclosed.Porch','X3Ssn.Porch','Screen.Porch','Pool.Area','Fence','Misc.Val','Mo.Sold','Yr.Sold','HasShed','HasAlley','Garage.Age','Remod.Age','House.Age','TotalSF','TotalBath','All.Porch.SF','TotalSFLog','All.SF','AllSF-s2','GrLivArea-s2','MS.SubClass_30','MS.SubClass_50','MS.SubClass_60','MS.SubClass_70','MS.SubClass_80','MS.SubClass_85','MS.SubClass_90','MS.SubClass_120','MS.SubClass_160','MS.SubClass_190','MS.SubClass_Other','MS.Zoning_RH','MS.Zoning_RL','MS.Zoning_RM','Land.Contour_HLS','Land.Contour_Low','Land.Contour_Lvl','Lot.Config_CulDSac','Lot.Config_FR2','Lot.Config_FR3','Lot.Config_Inside','Neighborhood_BrDale','Neighborhood_BrkSide','Neighborhood_ClearCr','Neighborhood_CollgCr','Neighborhood_Crawfor','Neighborhood_Edwards','Neighborhood_Gilbert','Neighborhood_IDOTRR','Neighborhood_MeadowV','Neighborhood_Mitchel','Neighborhood_NAmes','Neighborhood_NPkVill','Neighborhood_NWAmes','Neighborhood_NoRidge','Neighborhood_NridgHt','Neighborhood_OldTown','Neighborhood_SWISU','Neighborhood_Sawyer','Neighborhood_SawyerW','Neighborhood_Somerst','Neighborhood_StoneBr','Neighborhood_Timber','Neighborhood_Veenker','Bldg.Type_2fmCon','Bldg.Type_Duplex','Bldg.Type_Twnhs','Bldg.Type_TwnhsE','House.Style_1.5Unf','House.Style_1Story','House.Style_2.5Fin','House.Style_2.5Unf','House.Style_2Story','House.Style_SFoyer','House.Style_SLvl','Roof.Style_Hip','Roof.Style_Other','Mas.Vnr.Type_Stone','Mas.Vnr.Type_Other','Mas.Vnr.Type_None','Foundation_CBlock','Foundation_PConc','Foundation_Other','Bsmt.Qual_Gd','Bsmt.Qual_TA','Bsmt.Qual_Fa','Bsmt.Qual_NA','Bsmt.Cond_TA','Bsmt.Cond_Fa','Bsmt.Cond_NA','Bsmt.Exposure_Av','Bsmt.Exposure_Mn','Bsmt.Exposure_No','Bsmt.Exposure_NA','BsmtFin.Type.1_ALQ','BsmtFin.Type.1_BLQ','BsmtFin.Type.1_Rec','BsmtFin.Type.1_LwQ','BsmtFin.Type.1_Unf','BsmtFin.Type.1_NA','BsmtFin.Type.2_ALQ','BsmtFin.Type.2_BLQ','BsmtFin.Type.2_Rec','BsmtFin.Type.2_LwQ','BsmtFin.Type.2_Unf','BsmtFin.Type.2_NA','Central.Air_Y','Garage.Type_Attchd','Garage.Type_Basment','Garage.Type_BuiltIn','Garage.Type_CarPort','Garage.Type_Detchd','Garage.Finish_RFn','Garage.Finish_Unf','Sale.Type_GroupedWD','Sale.Type_Other','Sale.Condition_AdjLand','Sale.Condition_Alloca','Sale.Condition_Family','Sale.Condition_Normal','Sale.Condition_Partial','Condition_Railroad','Condition_Roads','Condition_Positive','Condition_RoadsAndRailroad','Exterior_BrkFace','Exterior_CemntBd','Exterior_HdBoard','Exterior_MetalSd','Exterior_Plywood','Exterior_Stucco','Exterior_VinylSd','Exterior_Wd Sdng','Exterior_WdShing','Exterior_Other', 'SalesPrice']

    try:
        # Receber dados JSON
        data = request.get_json()
        
        # Converter dados JSON em DataFrame e salvar como CSV
        pd.DataFrame([data]).to_csv('data/raw/oxe.csv', index=False)
        
        
        subprocess.run(['python', 'notebooks/combined_script.py'])
        
        # Carregar dados processados
        processed_data = pd.read_csv('data/processed/ames_model_data.csv')
        model_data = processed_data.copy()

        categorical_columns = []
        ordinal_columns = []
        for col in model_data.select_dtypes('category').columns:
            if model_data[col].cat.ordered:
                ordinal_columns.append(col)
            else:
                categorical_columns.append(col)

        for col in ordinal_columns:
            codes, _ = pd.factorize(data[col], sort=True)
            model_data[col] = codes

        original_data = model_data['Exterior']
        encoded_data = pd.get_dummies(original_data, drop_first=True)

        aux_dataframe = encoded_data
        aux_dataframe['Exterior'] = original_data.copy()


        model_data = pd.get_dummies(model_data, drop_first=True)


  # Adicionando a coluna faltante e preenchendo com zeros
            
        for col in model_data.columns:
            if col not in colunas_finais:
                model_data.drop(col, axis=1, inplace=True)
                
        for col in colunas_finais:
            if col not in model_data.columns:
                model_data[col] = 0

        # Fazer previsões
        prediction = model.predict(model_data)
        
        return jsonify({'prediction': prediction.tolist()})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
