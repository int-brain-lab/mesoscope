from one.api import ONE
from iblutil.util import Bunch
from iblatlas.atlas import AllenAtlas
import numpy as np
import matplotlib.pyplot as plt
from collections  import Counter
from rastermap import Rastermap
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba, hsv_to_rgb, to_hex
import gc
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from scipy.stats import zscore
from matplotlib import gridspec
from brainbox.io.one import SessionLoader
from brainbox.behavior.wheel import interpolate_position
from brainbox.behavior.wheel import velocity_filtered

plt.ion()
one = ONE()

pth_meso = Path(one.cache_dir, 'meso')
pth_meso.mkdir(parents=True, exist_ok=True)


atlas = AllenAtlas()

adf = atlas.regions.to_df()
region_colors_dict = {
    row['acronym']: f"{row['hexcolor']}"  
    for _, row in adf.iterrows()}


def load_distinct_bright_colors(n=20, saturation=0.9, brightness=0.95):
    hues = np.linspace(0, 1, n, endpoint=False)
    hsv = np.stack([hues, np.full(n, saturation), np.full(n, brightness)], axis=1)
    rgb = hsv_to_rgb(hsv)
    hex_colors = [to_hex(c) for c in rgb]
    return hex_colors


def embed_meso(eid):

    '''
    Load and embed mesoscope data via rastermap for a given experiment ID (eid).
    Parameters:
    - eid: str, experiment ID
    eid = '71e53fd1-38f2-49bb-93a1-3c826fbe7c13', Sam's example

    query = 'field_of_view__imaging_type__name,mesoscope'
    eids = one.search(procedures='Imaging', django=query, query_type='remote')


    scaling: bool, whether to scale the data by percentile
    '''
    print('Loading mesoscope data for experiment ID:', eid)

    objects = ['mpci', 'mpciROIs', 'mpciROITypes', 'mpciStack']  

    fov_folders = one.list_collections(eid, collection='alf/FOV_*')
    fovs = sorted(map(lambda x: int(x[-2:]), fov_folders))
    nFOV = len(fovs)

    all_ROI_data = Bunch()
    for fov in fov_folders:
        all_ROI_data[fov.split('/')[-1]] = one.load_collection(eid, fov, object=objects)

    roi_signals = []
    roi_timess = []
    region_labelss = []
    region_colorss = []
    xyzs = []

    for fov in all_ROI_data:
        print(fov)
        ROI_data_00 = all_ROI_data[fov]

        # Determine region alignment key
        key = 'brainLocationsIds_ccf_2017' if 'brainLocationsIds_ccf_2017' in ROI_data_00['mpciROIs'] \
            else 'brainLocationIds_ccf_2017_estimate'
        
        region_ids = ROI_data_00['mpciROIs'][key]
        region_labels = atlas.regions.id2acronym(region_ids)


        region_colors = np.array([adf.loc[adf['id'] == i, 
                            'hexcolor'].values[0] for i in region_ids])

        # Data: times and ROI signals
        frame_times = ROI_data_00['mpci']['times']
        roi_xyz = ROI_data_00['mpciROIs']['stackPos']
        timeshift = ROI_data_00['mpciStack']['timeshift']
        roi_offsets = timeshift[roi_xyz[:, len(timeshift.shape)]]
        
        roi_times = np.tile(frame_times, 
                            (roi_offsets.size, 1)) + roi_offsets[np.newaxis, :].T

        # roi_signal = ROI_data_00['mpci']['ROIActivityF'].T  
        roi_signal = ROI_data_00['mpci']['ROIActivityDeconvolved'].T 
        
        print(roi_times.shape[1], 'time bins', 'from ', roi_times[0].min(), 
            'to', roi_times[0].max(), 
            f'bin size: {np.diff(roi_times[0])[0]:.4f} s')

        # filter out neurons only
        mask = ROI_data_00['mpciROIs']['mpciROITypes']
        mask = mask.astype(bool)

        print(fov, sum(mask), 'of', len(mask), 'channels are neurons')

        roi_signals.append(roi_signal[mask])
        roi_timess.append(roi_times)
        region_labelss.append(region_labels[mask])
        region_colorss.append(region_colors[mask])
        xyzs.append(ROI_data_00['mpciROIs']['mlapdv_estimate'][mask])

        
    # stack across fovs    
    roi_signal = np.vstack(roi_signals)
    roi_times = roi_timess[0]  # all fovs have the same time bins
    region_labels = np.hstack(region_labelss)
    region_colors = np.hstack(region_colorss)
    xyz = np.vstack(xyzs)

    print(roi_signal.shape, 'ROI signal shape')
    print(Counter(region_labels))


    print('Running rastermap...')
    model = Rastermap(n_PCs=100, n_clusters=30,
                    locality=0.75, time_lag_window=5, 
                    bin_size=1).fit(roi_signal)

    isort = model.isort

    rr = {
        'roi_signal': roi_signal,
        'roi_times': roi_times,
        'region_labels': region_labels,
        'region_colors': region_colors,
        'isort': isort,
        'xyz': xyz}

    dpth = Path(pth_meso, 'data')
    dpth.mkdir(parents=True, exist_ok=True)
    np.save(Path(dpth, f"{eid}.npy"), rr, allow_pickle=True)
    return rr

def load_or_embed(eid, restrict=None):
    fpath = Path(pth_meso, 'data', f"{eid}.npy")
    if fpath.exists():
        rr = np.load(fpath, allow_pickle=True).item()
    else:
        rr = embed_meso(eid)   # run your embedding function

    if restrict is not None:
        print('pre restrict shape:', rr['roi_signal'].shape)
        rr['roi_signal'] = rr['roi_signal'][restrict, :] 
        rr['region_labels'] = rr['region_labels'][restrict]
        rr['xyz'] = rr['xyz'][restrict]
        del rr['isort']  # no longer valid after restriction
        print('post restrict shape:', rr['roi_signal'].shape)
    return rr


def plot_raster(eid, bg='regions', alpha_bg=0.3, alpha_data=0.5, interp='none', 
                restr=True, rsort=True, scaling=True, restrict=None):
    '''
    restr: restrict to 1 min starting at end of first third of recording

    put wheel speed trace on top of rastermap
    '''

    rr = load_or_embed(eid, restrict=restrict)
    if restrict is not None:
        rsort = False  # no isort after restriction
        print('restricted to', len(rr['roi_signal']), 'cells')

    if rsort:
        # Sort the data by isort
        rr['roi_signal'] = rr['roi_signal'][rr['isort']]
        rr['region_labels'] = rr['region_labels'][rr['isort']]
        rr['region_colors'] = rr['region_colors'][rr['isort']]


    # Allen colors are too similar for these visual areas, remap to distinct colors
    regs = np.unique(rr['region_labels'])   
    region_colors_d = dict(zip(regs,load_distinct_bright_colors(n=len(regs))))
    region_colors = np.array([region_colors_d[reg] for reg in rr['region_labels']])

    if scaling:
        # scaling every trace between its 20th and 99th percentile
        print('Scaling ROI signals...')
        p20 = np.percentile(rr['roi_signal'], 20, axis=1, keepdims=True)
        p99 = np.percentile(rr['roi_signal'], 99, axis=1, keepdims=True)
        rr['roi_signal'] = (rr['roi_signal'] - p20) / (p99 - p20)


    n_rows, n_time = rr['roi_signal'].shape
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(10, 1, height_ratios=[1] + [1]*9, hspace=0.05)
    ax_wheel = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1:], sharex=ax_wheel)

    if restr:
        # Time resolution (samples per second)
        sampling_rate = rr['roi_times'][0].shape[0] / (rr['roi_times'][0].max() - rr['roi_times'][0].min())

        # Number of time bins in 1 minute
        n_1min = int(sampling_rate * 60)

        # Start of second third
        start = n_time // 3

        rr['roi_signal'] = rr['roi_signal'][:, start:start + n_1min]
        rr['roi_times'] = rr['roi_times'][:, start:start + n_1min]


    # get wheel speed
    # sess_loader = SessionLoader(one, eid)
    # sess_loader.load_wheel()
    # wheel = sess_loader.wheel


    wheel = one.load_object(eid, 'wheel')
    wh_pos_lin, w_times = interpolate_position(wheel['timestamps'], wheel['position'],freq=250)
    w_velo, _ = velocity_filtered(wh_pos_lin, 250)


    # Restrict to same time range
    t_min, t_max = rr['roi_times'][0].min(), rr['roi_times'][0].max()
    mask = (w_times >= t_min) & (w_times <= t_max)
    w_times = w_times[mask]
    w_velo = w_velo[mask]

    # Plot wheel velocity
    ax_wheel.plot(w_times, w_velo, color='black', linewidth=0.8)
    ax_wheel.set_ylabel('Wheel\nvelocity', fontsize=8)
    ax_wheel.spines['top'].set_visible(False)
    ax_wheel.spines['right'].set_visible(False)
    ax_wheel.tick_params(labelbottom=False, length=2, pad=2)   


    print(np.min(rr['roi_signal']), np.max(rr['roi_signal']), 
          'min and max of roi_signal')

    vmin, vmax = np.min(rr['roi_signal']), 0.1  # np.max(rr['roi_signal'])
    ax.imshow(rr['roi_signal'], cmap='gray_r', aspect='auto', interpolation=interp,
                    extent=[rr['roi_times'][0].min(), rr['roi_times'][0].max(), 0, n_rows], vmin=vmin, vmax=vmax,
                    zorder=1,alpha=alpha_data)

    if bg == 'regions':
        for i, color in enumerate(region_colors):
            ax.fill_between(
                [rr['roi_times'][0].min(), rr['roi_times'][0].max()],
                i, i + 1, facecolor=color, alpha=alpha_bg, linewidth=0, zorder=0
            )


        region_counter = Counter(rr['region_labels'])

        patches = [
            mpatches.Patch(color=region_colors_d[region], label=f"{region} ({count})")
            for region, count in region_counter.items()
        ]

        n_cols = len(patches)  # one column per region
        legend = ax.legend(
            handles=patches,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.12),
            ncol=min(6, n_cols if n_cols <= 3 else n_cols // 2),
            frameon=False,
            fontsize='small'
        )


    if bg == 'firing_rate':
        # 1. Compute firing rate per ROI (row)
        firing_rate_per_row = np.mean(rr['roi_signal'], axis=1)  # (n_rows,)

        # 2. Normalize to [0, 1]
        fr_norm = (firing_rate_per_row - firing_rate_per_row.min()) / (firing_rate_per_row.max() - firing_rate_per_row.min())

        # 3. Create 2D background image: repeat firing rates across time axis
        firing_rate_img = np.tile(fr_norm[:, np.newaxis], (1, rr['roi_signal'].shape[1]))

        # 4. Plot with imshow as background (under main signal)
        cmap_bg = plt.cm.inferno  # or any colormap you like

        im_bg = ax.imshow(firing_rate_img, cmap=cmap_bg, aspect='auto', interpolation='none',
                        extent=[rr['roi_times'][0].min(), rr['roi_times'][0].max(), 0, n_rows],
                        zorder=0)

        # 5. Add colorbar
        cbar = plt.colorbar(im_bg, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Mean firing rate per ROI', fontsize=10)


    ax.set_xlabel('Time [s]')
    ax.set_ylabel('rastermap sorted ROIs' if rsort else 'ROIs')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.colorbar(im, ax=ax, label='Activity (dF/F)')
    plt.tight_layout()
    fig.suptitle(f"{eid} restrict to {len(rr['roi_signal'])}" if restrict is not None else eid)
    fig.savefig(Path(pth_meso, 'rasters', 
        f"{eid}_{'_'.join(np.unique(rr['region_labels']))}_{'_'.join([str(x) for x in rr['roi_signal'].shape])}.png"), dpi=300)
    #plt.close()



def plot_xyz(eid, mapping='isort', axoff=False, ax=None, restrict=None):

    '''
    3d plot of cell locations
    '''

    r = load_or_embed(eid, restrict=restrict)

    alone = False
    if not ax:
        alone = True
        fig = plt.figure(figsize=(8.43,7.26), label=mapping)
        ax = fig.add_subplot(111,projection='3d')   


    xyz = r['xyz'] / 1000  # isorted xyz coordinates; in mm

    if restrict is not None:
        print('restricted to', len(r['roi_signal']), 'cells')
        print('no isort available after restriction -> mapping to regions')
        mapping = 'regions'  # no isort after restriction

    if mapping == 'isort':
        color_values = r['isort'] / r['isort'].max()
        cmap = cm.get_cmap('Spectral')

        sc = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], depthshade=False,
                        marker='o', s = 1 if alone else 0.5, c=color_values, cmap=cmap)

        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.1)
        cbar.set_label('isort rank (normalized)', fontsize=10)

    elif mapping == 'regions':
        regs = np.unique(r['region_labels'])   
        region_colors_d = dict(zip(regs, load_distinct_bright_colors(n=len(regs))))
        cols = np.array([region_colors_d[reg] for reg in r['region_labels']])

        ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], depthshade=False,
                marker='o', s = 1 if alone else 0.5, c=cols)


           
    scalef = 1                 
    ax.view_init(elev=45.78, azim=-33.4)
    ax.set_xlim(min(xyz[:,0])/scalef, max(xyz[:,0])/scalef)
    ax.set_ylim(min(xyz[:,1])/scalef, max(xyz[:,1])/scalef)
    ax.set_zlim(min(xyz[:,2])/scalef, max(xyz[:,2])/scalef)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    fontsize = 14
    ax.set_xlabel('x [mm]', fontsize = fontsize)
    ax.set_ylabel('y [mm]', fontsize = fontsize)
    ax.set_zlabel('z [mm]', fontsize = fontsize)
    ax.tick_params(axis='both', labelsize=12)
    #ax.set_title(f'Mapping: {mapping}')
    ax.grid(False)
    nbins = 3
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))

    if axoff:
        ax.axis('off')

    if restrict is not None:
        ss = f'color: {mapping}, restricted to {len(r["xyz"])} cells, \n eid = {eid}'
    else:
        ss = f'color: {mapping}, {len(r["xyz"])} cells \n eid = {eid}'
    ax.set_title(ss)

    # Add legend with region label names and colors
    if mapping == 'regions':
        handles = [
            mpatches.Patch(color=region_colors_d[reg], label=reg)
            for reg in sorted(regs)
        ]
        if alone:
            # Create a separate legend outside the 3D plot if standalone
            fig.subplots_adjust(right=0.75)
            ax.legend(handles=handles, loc='center left',
                      bbox_to_anchor=(1.05, 0.5),
                      fontsize='small', frameon=False)
        else:
            # Add legend inside current axes if ax was passed
            ax.legend(handles=handles, loc='upper right',
                      fontsize='small', frameon=False)



def deep_in_block(trials, pleft, depth=10):

    '''
    get mask for trials object of pleft trials that are 
    "depth" trials into the block
    '''
    
    # pleft trial indices 
    ar = np.arange(len(trials['stimOn_times']))[trials['probabilityLeft'] == pleft]
    
    # pleft trial indices shifted by depth earlier 
    ar_shift = ar - depth
    
    # trial indices where shifted ones are in block
    ar_ = ar[trials['probabilityLeft'][ar_shift] == pleft]

    # transform into mask for all trials
    bool_array = np.full(len(trials['stimOn_times']), False, dtype=bool)
    bool_array[ar_] = True
    
    return bool_array



def get_win_times(eid):

    trials = one.load_object(eid, 'trials')
    mask = np.full(len(trials['stimOn_times']), True, dtype=bool)  # replace at some point??

    # For the 'inter_trial' mask trials with too short iti        
    idcs = [0]+ list(np.where((trials['stimOn_times'][1:]
                - trials['intervals'][:,-1][:-1])>1.15)[0]+1)
    mask_iti = [True if i in idcs else False 
        for i in range(len(trials['stimOn_times']))]


    # {window_name: [alignment event, mask, [win_start, win_end]]}
    # win_start is negative relative to event, win_end is positive relative to event

    tts = {

    'inter_trial': ['stimOn_times',
                np.bitwise_and.reduce([mask, mask_iti]),
                [1.15, -1]],  
    'blockL': ['stimOn_times', 
                np.bitwise_and.reduce([mask, 
                trials['probabilityLeft'] == 0.8]), 
                [0.4, -0.1]],
    'blockR': ['stimOn_times', 
                np.bitwise_and.reduce([mask, 
                trials['probabilityLeft'] == 0.2]),
                [0.4, -0.1]],
    'block50': ['stimOn_times', 
                np.bitwise_and.reduce([mask, 
                trials['probabilityLeft'] == 0.5]),
                [0.4, -0.1]],                                            
    'quiescence': ['stimOn_times', mask, 
                [0.4, -0.1]],                       
    'stimLbLcL': ['stimOn_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastLeft']),
            trials['probabilityLeft'] == 0.8,
            deep_in_block(trials, 0.8),
            trials['choice'] == 1]), 
                                [0, 0.2]], 
    'stimLbRcL': ['stimOn_times',            
        np.bitwise_and.reduce([mask,
            ~np.isnan(trials[f'contrastLeft']), 
            trials['probabilityLeft'] == 0.2,                       
            deep_in_block(trials, 0.2),
            trials['choice'] == 1]), [0, 0.2]],
    'stimLbRcR': ['stimOn_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastLeft']), 
            trials['probabilityLeft'] == 0.2,
            deep_in_block(trials, 0.2),
            trials['choice'] == -1]), 
                                [0, 0.2]],           
    'stimLbLcR': ['stimOn_times',
            np.bitwise_and.reduce([mask,       
            ~np.isnan(trials[f'contrastLeft']), 
            trials['probabilityLeft'] == 0.8,
            deep_in_block(trials, 0.8),
            trials['choice'] == -1]), 
                                [0, 0.2]],
    'stimRbLcL': ['stimOn_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastRight']), 
            trials['probabilityLeft'] == 0.8,
            deep_in_block(trials, 0.8),
            trials['choice'] == 1]), 
                                [0, 0.2]], 
    'stimRbRcL': ['stimOn_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastRight']), 
            trials['probabilityLeft'] == 0.2,
            deep_in_block(trials, 0.2),
            trials['choice'] == 1]), 
                                [0, 0.2]],
    'stimRbRcR': ['stimOn_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastRight']), 
            trials['probabilityLeft'] == 0.2,
            deep_in_block(trials, 0.2),
            trials['choice'] == -1]), 
                                [0, 0.2]],        
    'stimRbLcR': ['stimOn_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastRight']), 
            trials['probabilityLeft'] == 0.8,
            deep_in_block(trials, 0.8),
            trials['choice'] == -1]), 
                                [0, 0.2]],
    'motor_init': ['firstMovement_times', mask, 
                [0.15, 0]],                                        
    'sLbLchoiceL': ['firstMovement_times',
            np.bitwise_and.reduce([mask,  
            ~np.isnan(trials[f'contrastLeft']), 
            trials['probabilityLeft'] == 0.8,
            trials['choice'] == 1]), 
                                [0.15, 0]], 
    'sLbRchoiceL': ['firstMovement_times',
        np.bitwise_and.reduce([mask,
            ~np.isnan(trials[f'contrastLeft']), 
            trials['probabilityLeft'] == 0.2,
            trials['choice'] == 1]), 
                                [0.15, 0]],
    'sLbRchoiceR': ['firstMovement_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastLeft']), 
            trials['probabilityLeft'] == 0.2,
            trials['choice'] == -1]), 
                                [0.15, 0]],           
    'sLbLchoiceR': ['firstMovement_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastLeft']), 
            trials['probabilityLeft'] == 0.8,
            trials['choice'] == -1]), 
                                [0.15, 0]],
    'sRbLchoiceL': ['firstMovement_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastRight']), 
            trials['probabilityLeft'] == 0.8,
            trials['choice'] == 1]), 
                                [0.15, 0]], 
    'sRbRchoiceL': ['firstMovement_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastRight']), 
                                trials['probabilityLeft'] == 0.2,
                                trials['choice'] == 1]), 
                                [0.15, 0]],
    'sRbRchoiceR': ['firstMovement_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastRight']), 
                                trials['probabilityLeft'] == 0.2,
                                trials['choice'] == -1]), 
                                [0.15, 0]],        
    'sRbLchoiceR': ['firstMovement_times',
            np.bitwise_and.reduce([mask, 
            ~np.isnan(trials[f'contrastRight']), 
                                trials['probabilityLeft'] == 0.8,
                                trials['choice'] == -1]), 
                                [0.15, 0]],
    'choiceL': ['firstMovement_times', 
        np.bitwise_and.reduce([mask,
            trials['choice'] == 1]), 
                [0, 0.15]],
    'choiceR': ['firstMovement_times', 
        np.bitwise_and.reduce([mask,
            trials['choice'] == -1]), 
                [0, 0.15]],            
    'fback1': ['feedback_times',    
        np.bitwise_and.reduce([mask,
            trials['feedbackType'] == 1]), 
                [0, 0.3]],
    'fback0': ['feedback_times', 
        np.bitwise_and.reduce([mask,
            trials['feedbackType'] == -1]), 
                [0, 0.3]]}

    return trials, tts


def sparseness(rvec):
    """
    Treves-Rolls population sparseness for a vector of nonnegative responses r_i.
    a_p = (mean(r))^2 / mean(r^2), with guards for degenerate cases.
    """
    r = np.asarray(rvec, dtype=float)
    if r.size == 0:
        return np.nan
    m1 = r.mean()
    m2 = np.mean(r * r)
    if m2 <= 0:
        return 0.0
    return (m1 * m1) / m2


def compute_sparseness(eid, scaling=True, restrict=None):
    '''
    For a given eid comput the sparseness; for the whole recording
    and all specific trial structure time windows
    '''
    rr = load_or_embed(eid, restrict=restrict)

    if scaling:
        # scaling every trace between its 20th and 99th percentile
        print('Scaling ROI signals...')
        p20 = np.percentile(rr['roi_signal'], 20, axis=1, keepdims=True)
        p99 = np.percentile(rr['roi_signal'], 99, axis=1, keepdims=True)
        scale = np.clip(p99 - p20, 1e-3, None)
        rr['roi_signal'] = (rr['roi_signal'] - p20) / scale
        rr['roi_signal'] = np.nan_to_num(rr['roi_signal'], nan=0.0, posinf=1.0, neginf=0.0)

    # NOTE: This uses the first ROI's time base as the global axis.
    # If per-ROI offsets matter for analysis, resample onto a shared grid.
    times = rr['roi_times'][0]
    trials, tts = get_win_times(eid)
    T = times.size

    res = {}
    # iterate through windows
    for tt in tts:
        event = trials[tts[tt][0]][tts[tt][1]]
        start, end = tts[tt][2]  # relative to event
        win_times = event[:, np.newaxis] - np.array([start, -end])
        
        # keep only valid windows (skip NaNs)
        valid = ~(np.isinf(win_times) | np.isnan(win_times)).all(axis=1)
        w = win_times[valid]
        starts, ends = w[:, 0], w[:, 1]

        # clip windows to the recorded interval
        starts = np.clip(starts, times[0], times[-1])
        ends   = np.clip(ends,   times[0], times[-1])

        # indices that include all samples within each window
        # start is inclusive, end is exclusive (classic slice semantics)
        idx_start = np.searchsorted(times, starts, side='left')
        idx_end_excl = np.searchsorted(times, ends,   side='right')  # exclusive

        # ---- build one boolean mask for all windows efficiently ----
        mask = np.zeros(T + 1, dtype=int)
        np.add.at(mask, idx_start,  1)
        np.add.at(mask, idx_end_excl, -1)
        mask = np.cumsum(mask)[:T] > 0  # shape (T,), True inside any window

        # select data inside windows (inclusive of boundary times)
        sig_in_windows = rr['roi_signal'][:, mask]  # shape (N_neurons, T_in_windows)
        res[tt] = {}
        res[tt]['n_trials'] = w.shape[0]
        res[tt]['n_time_bins'] = mask.sum()
        res[tt]['firing_rates'] = sig_in_windows.mean(axis=1)
        res[tt]['sparseness'] = sparseness(res[tt]['firing_rates'])

    res['full_session'] = {}
    res['full_session']['n_trials'] = trials['stimOn_times'].size
    res['full_session']['n_time_bins'] = T
    res['full_session']['firing_rates'] = rr['roi_signal'].mean(axis=1)
    res['full_session']['sparseness'] = sparseness(
        res['full_session']['firing_rates'])
    
    dpth = Path(pth_meso, 'sparseness')
    dpth.mkdir(parents=True, exist_ok=True)
    if restrict is not None:
        s = f"{eid}_restricted.npy"
    else:
        s = f"{eid}.npy"
    np.save(Path(dpth, s), res, allow_pickle=True)
    return res


def load_or_compute_sparseness(eid, restrict=None):
    if restrict is not None:
        fpath = Path(pth_meso, 'sparseness', f"{eid}_restricted.npy")
    else:
        fpath = Path(pth_meso, 'sparseness', f"{eid}.npy")

    if fpath.exists():
        res = np.load(fpath, allow_pickle=True).item()
    else:
        print('Computing sparseness for eid:', eid)
        res = compute_sparseness(eid, restrict=restrict)   # run your embedding function
    return res


def plot_sparseness_res(eid, save=True, show=True, restrict=None):
    """
    For a given session, plot a 4-panel *vertical* bar summary:
      1) n_trials
      2) n_time_bins
      3) mean firing rate across neurons
      4) sparseness (Treves–Rolls)
    Includes all time windows plus 'full_session' (where n_trials is NaN).
    Bars are black. All panels share the same x (window labels).
    """
    res = load_or_compute_sparseness(eid, restrict=restrict)

    # keep insertion order, but put 'full_session' last
    keys = list(res.keys())
    if 'full_session' in keys:
        keys = [k for k in keys if k != 'full_session'] + ['full_session']

    # collect series
    n_trials, n_bins, fr_mean, spars = [], [], [], []
    for k in keys:
        d = res[k]
        n_trials.append(float(d.get('n_trials', np.nan)))
        n_bins.append(int(d['n_time_bins']))
        fr = d.get('firing_rates', None)
        fr_mean.append(float(np.nanmean(fr)) if fr is not None and np.size(fr) else np.nan)
        spars.append(float(d.get('sparseness', np.nan)))

    x = np.arange(len(keys))

    # sizing: width scales with number of windows; height fixed for 4 rows
    fig_w = 8
    fig_h = 9.0
    fig, axes = plt.subplots(4, 1, figsize=(fig_w, fig_h), sharex=True)

    bar_kwargs = dict(color='black', width=0.8)

    # Row 1: n_trials
    axes[0].bar(x, n_trials, **bar_kwargs)
    axes[0].set_ylabel('n_trials', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)

    # Row 2: n_time_bins
    axes[1].bar(x, np.log(n_bins), **bar_kwargs)
    axes[1].set_ylabel('log(n_time_bins)', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)

    # Row 3: mean firing rate across neurons
    axes[2].bar(x, fr_mean, **bar_kwargs)
    axes[2].set_ylabel('mean firing rate', fontsize=11)
    axes[2].grid(axis='y', alpha=0.3)

    # Row 4: sparseness
    axes[3].bar(x, spars, **bar_kwargs)
    axes[3].set_ylabel('sparseness (Treves–Rolls)', fontsize=11)
    axes[3].grid(axis='y', alpha=0.3)

    # shared x: ticks/labels only on bottom axis
    for ax in axes:
        ax.set_xlim(-0.5, len(keys) - 0.5)
    for ax in axes[:-1]:
        ax.tick_params(axis='x', labelbottom=False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(keys, rotation=55, ha='right')

    if restrict is not None:
        s = f"eid {eid}, restricted to {len(res['full_session']['firing_rates'])} cells" 
    else:
        s = f"eid {eid}, {len(res['full_session']['firing_rates'])}  cells"

    fig.suptitle(s, fontsize=12, y=0.995)
    fig.tight_layout()

    if save:
        outdir = Path(pth_meso, 'sparseness')
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{eid}_sparseness_summary_vertical.png"
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Saved: {outfile}")

    if show:
        plt.show()
    else:
        plt.close(fig)




# if __name__ == "__main__":
#     # Query for all mesoscope experiments
#     query = 'field_of_view__imaging_type__name,mesoscope'
#     eids = one.search(procedures='Imaging', django=query, query_type='remote')

#     print(f"Found {len(eids)} mesoscope experiment IDs.")

#     for eid in eids:
#         try:
#             print(f"\nProcessing {eid}")
#             embed_meso(eid)
#             plot_meso(eid)
#         except Exception as e:
#             print(f"Failed to process {eid}: {e}")


                      
