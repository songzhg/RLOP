def csv_to_line_chart(path, single_chart = False, delimiter='\t'):
    import pandas as pd
    # import matplotlib
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    data = pd.read_csv(path, delimiter=delimiter)
    x = data.iloc[:, 0]

    plt.rcParams.update({'font.size': 18}) 
    
    if single_chart:
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        for col in data.columns[1:]:
            plt.plot(x, data[col], label=col)
        plt.legend()
    else:
        num_cols = data.shape[1] - 1
        rows = max(1, int(num_cols ** 0.5))
        cols = max(1, (num_cols + rows - 1) // rows) 
        fig, axs = plt.subplots(rows, cols, figsize=(15, 15), squeeze=False)
        axs = axs.flatten()
        for i in range(1, data.shape[1]):
            ax = axs[i - 1]
            ax.plot(x, data.iloc[:, i], label=data.columns[i])
            ax.set_xlabel(data.columns[0]) 
            ax.set_ylabel(data.columns[i])
            ax.legend()
        for j in range(i, len(axs)):
            axs[j].axis('off')
        plt.tight_layout()
    plt.show()

def tensorboard_to_csv(path):
    from tensorboard.backend.event_processing import event_accumulator
    import csv
    import os
    ea = event_accumulator.EventAccumulator(
        path, size_guidance={ 
            event_accumulator.SCALARS: 0, 
        }
    )
    ea.Reload()
    scalar_tags = ea.Tags()['scalars']
    scalars_data = {}
    for tag in scalar_tags:
        scalars_data[tag] = ea.Scalars(tag)
    max_length = max(len(data) for data in scalars_data.values())
    output_path = os.path.join(path, 'log.csv')
    print(output_path)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        header = ['step'] + scalar_tags
        writer.writerow(header)
        for i in range(max_length):
            row = [ scalars_data[scalar_tags[0]][i].step ]
            for tag in scalar_tags:
                value = scalars_data[tag][i].value if i < len(scalars_data[tag]) else None
                row.append(value)
            writer.writerow(row)

def plot_rlop_sb3(rlop_path, sb3_path, x_label, y_label, delimiter='\t'):
    import pandas as pd
    import matplotlib.pyplot as plt

    rlop_data = pd.read_csv(rlop_path, delimiter=delimiter)
    sb3_data = pd.read_csv(sb3_path, delimiter=delimiter)
    
    rlop_x = list(rlop_data[x_label['rlop']].to_numpy())
    rlop_y = list(rlop_data[y_label['rlop']].to_numpy())
    sb3_x = list(sb3_data[x_label['sb3']].to_numpy())
    sb3_y = list(sb3_data[y_label['sb3']].to_numpy())

    plt.figure(figsize=(12, 8))
    plt.plot(sb3_x, sb3_y, marker='s', label='sb3')
    plt.plot(rlop_x, rlop_y, marker='o', label='rlop')
    plt.title('Compare')
    plt.xlabel(x_label['rlop'])
    plt.ylabel(y_label['rlop'])
    plt.legend()
    plt.show()

# idx = 0
# tensorboard_to_csv("data/dqn/lunar_lander/sb3/DQN_" + str(idx + 1))
# rlop_path = "data/dqn/lunar_lander/rlop_" + str(idx) + "_log.txt"
# sb3_path = "data/dqn/lunar_lander/sb3/DQN_" + str(idx + 1) + "/log.csv"
# plot_rlop_sb3(rlop_path, sb3_path, x_label = { 'rlop': 'time_steps', 'sb3': 'step'}, y_label = { 'rlop': 'loss', 'sb3': 'train/loss'})