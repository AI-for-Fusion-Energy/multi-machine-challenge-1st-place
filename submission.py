
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------

fname = 'predictions.csv'
print('Reading:', fname)
predictions = pd.read_csv(fname, index_col=0)

# -----------------------------------------------------------------------------

selected_columns = ['.tci.results:nl_04',
                    'Greenwald_fraction',
                    '\\AXA.chord_9',
                    '\\AXJ.chord_23',
                    '\\AXJ.chord_3',
                    '\\MAGNETICS::TOP.ACTIVE_MHD.SIGNALS:BP02_GHK',
                    '\\MAGNETICS::TOP.ACTIVE_MHD.SIGNALS:BP13_GHK',
                    '\\MAGNETICS::TOP.ACTIVE_MHD.SIGNALS:BP20_ABK',
                    '\\btor',
                    '\\efit_aeqdsk:kappa',
                    '\\efit_aeqdsk:li',
                    '\\efit_aeqdsk:q95',
                    '\\efit_aeqdsk:rmagx',
                    '\\efit_aeqdsk:vloopt',
                    '\\efit_aeqdsk:wplasm',
                    '\\efit_aeqdsk:zmagx',
                    '\\xtomo::top.brightnesses.array_1:chord_38',
                    '\\xtomo::top.brightnesses.array_3:chord_02',
                    '\\xtomo::top.brightnesses.array_3:chord_14',
                    '\\xtomo::top.brightnesses.array_3:chord_33',
                    '\\xtomo::top.brightnesses.array_3:chord_36',
                    'n1_norm']

# -----------------------------------------------------------------------------

submission = []

for idx, row in predictions[selected_columns].iterrows():
    shot_id = 'ID_' + str(idx)
    shot_pred = int(np.round(row.mean()))
    submission.append([shot_id, shot_pred])

# -----------------------------------------------------------------------------

submission = pd.DataFrame(submission, columns=['Shot_list', 'Is_disrupt'])

fname = 'submission.csv'
print('Writing:', fname)
submission.to_csv(fname, index=False)
