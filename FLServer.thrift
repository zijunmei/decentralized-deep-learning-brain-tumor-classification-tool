/*
Federated Learning Server
- Clients request updated weights
- Aggregates new weights
*/

service FLServer
{
    list<list<list<list<double>>>> send_first_layer(),
    list<double> send_second_layer(),
    list<list<list<list<double>>>> send_third_layer(),
    list<double> send_fourth_layer(),
    list<list<double>> send_fifth_layer(),
    list<double> send_sixth_layer(),

    void receive_first_layer(1:string client_name, 2:list<list<list<list<double>>>> weights),
    void receive_second_layer(1:string client_name, 2:list<double> weights),
    void receive_third_layer(1:string client_name, 2:list<list<list<list<double>>>> weights),
    void receive_fourth_layer(1:string client_name, 2:list<double> weights),
    void receive_fifth_layer(1:string client_name, 2:list<list<double>> weights),
    void receive_sixth_layer(1:string client_name, 2:list<double> weights),

    i64 number_weights_collected(),
    i64 version_number(),
    i64 update_weights()
}