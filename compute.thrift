struct Model {
    1: list<list<double>> W
    2: list<list<double>> V
}

service Compute {
    Model get_gradient(),
    void train(1:string training_file),
    void set_model(1:Model shared_model)
}