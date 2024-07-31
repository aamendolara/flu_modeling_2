import pandas as pd
import matplotlib.pyplot as plt


## probably figure one here. I don'y know yet if I want to seperate each state or overlay them. I think seperate. 


vermont_raw = pd.read_csv('C:/Users/Alfred/OneDrive - Noorda College of Osteopathic Medicine/Flu-modeling/Vermont/Vermont_Data_Combined.csv')

hawaii_raw = pd.read_csv('C:/Users/Alfred/OneDrive - Noorda College of Osteopathic Medicine/Flu-modeling/Hawaii/Combined_Hawaii_Data.csv')

nevada_raw = pd.read_csv('C:/Users/Alfred/OneDrive - Noorda College of Osteopathic Medicine/Flu-modeling/Nevada/Nevada_Data_Combined.csv')

fig, axes = plt.subplots(3, sharex=True)
axes[0].plot("%UNWEIGHTED ILI", data = vermont_raw, color = 'purple', label = "Vermont")
axes[1].plot("%UNWEIGHTED ILI", data = hawaii_raw, color = 'green', label = "Hawaii")
plt.ylabel('% Unweighted ILI')
axes[2].plot("%UNWEIGHTED ILI", data = nevada_raw, color = 'yellow', label = "Nevada")
plt.xlabel('Week')
plt.show()
fig.savefig("figure_1", format = 'svg', dpi = 600)
