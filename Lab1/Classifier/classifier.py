import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here
    ham_list = file_lists_by_category[1]
    spam_list = file_lists_by_category[0]
    total_list = file_lists_by_category[1]+file_lists_by_category[0]

    # get word frequency in ham, spam and total
    ham_dic = util.get_word_freq(ham_list)
    spam_dic = util.get_word_freq(spam_list)
    total_dic = util.get_word_freq(total_list)
    
    #get total words in ham, spam and total
    N_spam_bag = sum(spam_dic.values())
    N_ham_bag = sum(ham_dic.values())

    #get feature dimension
    D_total_distinct_word = len(total_dic)
    
    #calculate p_d q_d
    p_d = {}
    q_d = {}

    for word in total_dic:
        word_count_in_spam = 0
        if(word in spam_dic):
            word_count_in_spam = spam_dic[word]
        p_d[word] = (word_count_in_spam + 1) / (N_spam_bag + D_total_distinct_word)
    
    for word in total_dic:
        word_count_in_ham = 0
        if(word in ham_dic):
            word_count_in_ham = ham_dic[word]
        q_d[word] = (word_count_in_ham + 1) / (N_ham_bag + D_total_distinct_word)
    
    probabilities_by_category = (p_d, q_d)   
    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    classify_result = tuple()
    
    #get the word frequency in file to be classified
    filename_list = (filename, 1)
    words_freq_dic = util.get_word_freq(filename_list[0:1])
    #print(words_freq_dic)

    #get the probabilities_by_category
    pd = probabilities_by_category[0]
    qd = probabilities_by_category[1]

    log_likelyhood_of_spam = np.log(prior_by_category[0])
    log_likelyhood_of_ham = np.log(prior_by_category[1])

    # calculate likelyhood of spam
    for word in words_freq_dic:
        freq = words_freq_dic[word]
        if(word in pd):
            pd_w = pd[word]
            log_likelyhood_of_spam += np.log(pd_w)*freq
        if(word in qd):
            qd_w = qd[word]     
            log_likelyhood_of_ham +=  np.log(qd_w)*freq

    res = "ham"
    if log_likelyhood_of_spam > log_likelyhood_of_ham:
        res = "spam"
    classify_result = (res,[log_likelyhood_of_spam,log_likelyhood_of_ham])
    
    return classify_result    

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)

    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    Type1_L = []
    Type2_L = []
    ratio_L = [1E-30, 1E-15, 1E-10, 1E-5, 1E0, 20, 50, 90, 1E2, 1E3]

    for r in ratio_L:
        performance_measures = np.zeros([2,2])
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                    probabilities_by_category,
                                                    priors_by_category)
            
            # Measure performance (the filename indicates the true label)
            log_spam_likelyhood = log_posterior[0]
            log_ham_likelyhood = log_posterior[1]

            #get true index
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            
            #recalculate guessed index and label
            label = "ham"
            if (log_spam_likelyhood + np.log(r) > log_ham_likelyhood):
                label = "spam"
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template="ratio = %f You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (r,correct[0],totals[0],correct[1],totals[1]))
        Type1_error = totals[0] - correct[0]
        Type2_error = totals[1] - correct[1]

        Type1_L.append(Type1_error)
        Type2_L.append(Type2_error)
    
    # plot type1 vs type2 error
    plt.plot(Type1_L, Type2_L)
    plt.xlabel('Type 1 Error count')
    plt.ylabel('Type 2 Error count')
    plt.title('Type 1 vs Type 2 Error trade off')
    plt.savefig("nbc.pdf")
    plt.show()
   

 