
def csv_to_line_chart(path, single_chart = False, delimiter='\t'):
    import pandas as pd
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
    # plt.savefig(path + '.pdf', bbox_inches='tight')
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
            row = [ i ]
            # row = [ scalars_data[scalar_tags[0]][i].step ]
            for tag in scalar_tags:
                # Retrieve the value if the current step exists for this tag
                value = scalars_data[tag][i].value if i < len(scalars_data[tag]) else None
                row.append(value)
            writer.writerow(row)


if __name__ == '__main__':
    # path = 'data/continuous_lunar_lander/'
    path = 'data/lunar_lander/'
    # tensorboard_to_csv(path)
    # csv_to_line_chart(path + '/rlop_ppo_log.txt', True)
    csv_to_line_chart(path + '/rlop_dqn_log.txt', True)
    # csv_to_line_chart(path + '/compare.txt', True)