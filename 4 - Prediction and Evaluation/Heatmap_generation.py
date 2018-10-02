import numpy as np
import matplotlib.pyplot as plt

pred_dim_dir = '/home/wli/Downloads/pred/dimensions/'
pred_dim_paths = glob.glob(osp.join(pred_dim_dir, '*.npy'))
pred_dim_paths.sort()

pred_rawheatmap_dir = '/home/wli/Downloads/pred/rawheatmap/'
pred_rawheatmap_paths = glob.glob(osp.join(pred_rawheatmap_dir, '*.npy'))
pred_rawheatmap_paths.sort()

for i in range(len(pred_dim_paths)):
    pred_dim = np.load(pred_dim_paths[i])
    pred_heatmap = np.load(pred_rawheatmap_paths[i])
    
    if pred_dim[7]*pred_dim[8] == len(pred_heatmap)*32:
        heatmap_final = pred_heatmap.reshape(pred_dim[7], pred_dim[8])
        
    else:
        
        heatmap_new = pred_heatmap[:pred_dim[7]*pred_dim[8]]
        heatmap_final = heatmap_new.reshape(pred_dim[7], pred_dim[8])
        
        
    heatmap_final_final = np.zeros((pred_dim[0], pred_dim[1]), heatmap_final.dtype)

    heatmap_final_final[:]=0.2722071

    heatmap_final_final[pred_dim[5]-1:pred_dim[6], pred_dim[3]-1:pred_dim[4]] = heatmap_final
    
    np.save('/home/wli/Downloads/pred/realheatmap_bbox/%s' % (osp.splitext(osp.basename(pred_dim_paths[i]))[0]), heatmap_final)
    np.save('/home/wli/Downloads/pred/realheatmap/%s' % (osp.splitext(osp.basename(pred_dim_paths[i]))[0]), heatmap_final_final)
