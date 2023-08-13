
predictions = model.predict(TestInput)


plt.figure(figsize=(20, 5))
for i in range(540, 550):
    
    plt.subplot(1, 10, i-539)
    plt.scatter(TrainInput[i, :, 0], TrainInput[i, :, 1], c= n_TrainInput[i, :, 2], s=5, cmap='jet')
    plt.title(str(np.where(n_TestInput[i]==1)[0]))
    plt.axis('scaled')
    plt.colorbar(shrink=0.25)
    plt.xticks([])
    plt.yticks([])
    
plt.show()


