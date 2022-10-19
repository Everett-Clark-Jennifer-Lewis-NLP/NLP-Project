def count(df):
    '''Plots a barplot of the distributions of character lengths and word counts within the repository readmes'''
    plt.figure(figsize=(16,9))
    plt.bar(range(len(df.word_count)), sorted(df.word_count), color='black')
    plt.bar(range(len(df.length)), sorted(df.length), color='lightsteelblue', alpha=0.5)
    plt.ylabel('Count')
    plt.xlim(0,100)
    plt.grid()
    plt.title('Distribution of Character Length and word Count per Repository')
    plt.legend(['Word Count', 'Length'])
    plt.show()