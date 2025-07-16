if __name__ == '__main__':
    print("数据路径:", opt.data_path)
    print("路径是否存在:", os.path.exists(opt.data_path))
    
    # 加载数据集
    dataset = dataset.Dataset_Load(data_path=opt.data_path, transform=dataset.ToTensor())
    print("数据集样本数:", len(dataset))  # 关键调试信息