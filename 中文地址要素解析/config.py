Config = {
    "tokenizer_path" : '/data/finetune_demo2/test/bert_gen/bert_base_chinese',
    "train_data_path": '/data/finetune_demo2/test/chinese_location_ner/dataset/train.txt',
    "train_data_path_all": '/data/finetune_demo2/test/chinese_location_ner/dataset/train_all.txt',
    "valid_data_path": '/data/finetune_demo2/test/chinese_location_ner/dataset/dev.txt',
    "test_data_path":'/data/finetune_demo2/test/chinese_location_ner/dataset/final_test.txt',
    "bert_path":'/data/finetune_demo2/test/bert_sft/bert_finetuned/checkpoint-20000',
    "model_path_all":'/data/finetune_demo2/test/chinese_location_ner/models/models_32_5e-5_60_linear_crf_bert_fintune_all.pth',
    "output_result_path":'/data/finetune_demo2/test/chinese_location_ner/result/output.txt',
    "epoch": 60,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "lr_scheduler_mod":"linear"
}
