import visualkeras
from utils.constants import *
from utils.regressor_tools import *
from utils.experiments_tools import *
import numpy as np
from PIL import ImageFont

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 32)

if __name__ == '__main__':
    input_dataset = read_csv_dataset(rrl_path, rrls_number)
    ids_dev = input_dataset[source_id].to_numpy().astype(str)
    periods_input = input_dataset[period].to_numpy()
    X, _, _ = read_time_series(ids_dev, data_path, max_rrl_point, periods=periods_input, max_phase=1.0)

for num, regressor_name in enumerate(regressors):
    regressor = create_regressor(regressor_name, X.shape[1:], '.', 1)
    print(regressor)
    visualkeras.layered_view(regressor.get_model(), to_file='output/model' + regressor_name + '.png', 
                            type_ignore=[visualkeras.SpacingDummyLayer], legend=True, font=font)
