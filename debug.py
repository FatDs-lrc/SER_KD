from models.networks.classifier import BertClassifier


teacher_path = '/data4/lrc/movie_dataset/pretrained/bert_movie_model'
net_teacher = BertClassifier.from_pretrained(teacher_path, num_classes=5, embd_method='max')
print(net_teacher)