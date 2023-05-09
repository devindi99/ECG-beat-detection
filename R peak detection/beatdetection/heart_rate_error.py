import numpy as np
from code_DataPrepare.data_prepare import get_episodes

def movmean(arr, window=3):
    arr = arr.reshape((-1,))
    paddedarr = np.pad(arr, (int(window/2), int(window/2)), 'edge')
    ret = np.cumsum(paddedarr, dtype=np.float32)  # if changed to float16 it gives NaN values
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

def hre_analysis(ref_annot_ori, test_annot_ori):
    fs = 250
    ref_peaks = ref_annot_ori[:, 0]
    test_peaks = test_annot_ori[:, 0]

    # refnoise = np.logical_or(ref_annot_ori[:, 6] == 1, ref_annot_ori[:, 5] == 1)
    # refnoise_episodes = get_episodes(refnoise)
    # testnoise = test_annot_ori[:, 6] == 1
    # testnoise_episodes = get_episodes(testnoise)
    # remove_episodes = np.zeros((refnoise_episodes.shape[0] + testnoise_episodes.shape[0], 2), dtype=int)
    # for i in range(0, refnoise_episodes.shape[0]):
    #     remove_episodes[i, 0] = ref_annot_ori[refnoise_episodes[i, 0], 0]
    #     remove_episodes[i, 1] = ref_annot_ori[refnoise_episodes[i, 1], 0]
    # for i in range(0, testnoise_episodes.shape[0]):
    #     remove_episodes[refnoise_episodes.shape[0] + i, 0] = test_annot_ori[testnoise_episodes[i, 0], 0]
    #     remove_episodes[refnoise_episodes.shape[0] + i, 1] = test_annot_ori[testnoise_episodes[i, 1], 0]
    # ref_test_noise = np.zeros((max(test_annot_ori[-1, 0], ref_annot_ori[-1, 0]), 1), dtype=int)
    # for i in range(remove_episodes.shape[0]):
    #     ref_test_noise[remove_episodes[i, 0]:remove_episodes[i, 1], 0] = 1

    ref_rr = np.diff(ref_peaks)
    ref_remidx = np.where(np.logical_or(ref_rr > 600, ref_rr < 60))[0] # HR between 25 and 250
    ref_rr = np.delete(ref_rr, ref_remidx, axis=0)
    ref_avg_rr = movmean(ref_rr, 3)
    ref_avg_hr = 15000/ref_avg_rr
    ref_x = np.asarray([x for x in range(int(fs/2), int(ref_peaks[-1]), int(fs/2)) ])
    ref_peaks = np.delete(ref_peaks, ref_remidx+1, axis=0)
    ref_interp_hr = np.interp(ref_x, ref_peaks[1:], ref_avg_hr)

    test_rr = np.diff(test_peaks)
    test_remidx = np.where(np.logical_or(test_rr > 600, test_rr < 60))[0]  # HR between 25 and 250
    test_rr = np.delete(test_rr, test_remidx, axis=0)
    test_avg_rr = movmean(test_rr, 3)
    test_avg_hr = 15000/test_avg_rr
    test_x = np.asarray([x for x in range(int(fs/2), int(test_peaks[-1]), int(fs/2))])
    test_peaks = np.delete(test_peaks, test_remidx + 1, axis=0)
    test_interp_hr = np.interp(test_x, test_peaks[1:], test_avg_hr)

    length = min(ref_interp_hr.shape[0], test_interp_hr.shape[0])
    sq_sum, hr_sum, remove = 0, 0, 0
    for i in range(1, length):
        if(ref_interp_hr[i] <= 20 or ref_interp_hr[i] >= 250 or test_interp_hr[i] <= 20 or test_interp_hr[i] >= 250):
            remove += 1
            continue
        a = ref_interp_hr[i] - test_interp_hr[i]
        sq_sum += a ** 2
        hr_sum += ref_interp_hr[i]
    # hre_norm = np.sqrt(sq_sum)/hr_sum
    # hre_unnorm = np.sqrt(sq_sum)/length

    hre_data = np.asarray([sq_sum, hr_sum, length-remove])

    return hre_data

def hre_calculation(sq_sum, hr_sum, length):
    # hre_norm = np.sqrt(sq_sum) / hr_sum
    # hre_unnorm = np.sqrt(sq_sum) / length
    hre_norm = 100 * np.sqrt(sq_sum * length) / hr_sum
    hre_unnorm = np.sqrt(sq_sum / length)
    ref_mean = hr_sum / length
    return hre_norm, hre_unnorm, ref_mean

def do_hre_analysis(hre_matrix_db, recordlist):
    assert hre_matrix_db.shape[0] == len(recordlist)

    db_table = []
    for rec in range(hre_matrix_db.shape[0]):
        hre_matrix = hre_matrix_db[rec]
        sq_sum, hr_sum, length = hre_matrix
        # hre_norm = np.sqrt(sq_sum)/rr_sum
        # hre_unnorm = np.sqrt(sq_sum)/length
        hre_norm, hre_unnorm, ref_mean = hre_calculation(sq_sum, hr_sum, length)

        row = [recordlist[rec], hre_unnorm, hre_norm, ref_mean]
        db_table.append(row)

    # Gross HRE measurements
    hre_matrix_sum = np.sum(hre_matrix_db, axis=0)
    sq_sum, hr_sum, length = hre_matrix_sum
    # hre_norm_gross = np.sqrt(sq_sum) / rr_sum
    # hre_unnorm_gross = np.sqrt(sq_sum) / length
    hre_norm_gross, hre_unnorm_gross, ref_mean_gross = hre_calculation(sq_sum, hr_sum, length)

    # Average HRE measurements
    arr_table = np.asarray(db_table)
    slice_table = arr_table[:, 1:].astype(float)
    assert slice_table.shape[1] == 3
    avg_table = np.nanmean(slice_table, axis=0)

    # Prepare final table
    gross_row = ['Gross', hre_unnorm_gross, hre_norm_gross, ref_mean_gross]
    avg_row = ['Average', avg_table[0], avg_table[1], avg_table[2]]

    db_table.append(gross_row)
    db_table.append(avg_row)

    return np.asarray(db_table)

if __name__ == '__main__':
    a = np.asarray([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    b = movmean(a, window=3)
    b = 9