import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, unpad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    return device

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=-1)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=-1)

  return xx_pad, yy_pad

def pad_collate_with_pack(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=-1)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=-1)

  return xx_pad, yy_pad, x_lens, y_lens

class LSTM2(torch.nn.ModuleList):
    def __init__(self, input_size, hidden_size, output_size, batch_size=64):
        super(LSTM2, self).__init__()
        # Set the hidden size, number of layers, and device for the model
        self.hidden_size = hidden_size
        self.device = device
        self.batch_size = batch_size
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
            dropout=0.1
        ).to(device)
        self.linear = torch.nn.Linear(hidden_size, output_size).to(device)
    
    def forward(self,x, x_lens):
        outputs = []
        #h_t = torch.zeros(1, self.batch_size, self.hidden_size, dtype=torch.float32)
        #c_t = torch.zeros(1, self.batch_size, self.hidden_size, dtype=torch.float32)  
        packed_output, (h_t, c_t) = self.lstm(x)
        # unpack the output
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        outputs = unpad_sequence(outputs, x_lens, batch_first=True)
        returned_outputs = []
        for tensor in outputs:
            returned_outputs.append(torch.sigmoid(self.linear(tensor)))
        return returned_outputs
    

class LSTM(torch.nn.ModuleList):
    def __init__(self, input_size, hidden_size, output_size, batch_size=64):
        super(LSTM, self).__init__()
        # Set the hidden size, number of layers, and device for the model
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = torch.nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )
        self.batch_size = batch_size
        self.linear = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x, x_lens):
        outputs = []
        x = unpad_sequence(x, x_lens, batch_first=True)
        h_ts = [
            torch.zeros(self.hidden_size, dtype=torch.float32).to(device)
            for _ in range(self.batch_size)
        ]
        c_ts = [
            torch.zeros(self.hidden_size, dtype=torch.float32).to(device)
            for _ in range(self.batch_size)
        ]
        for i,student in enumerate(x):
            for time_step in student:
                if time_step[0] == -1:
                    continue
                h_ts[i], c_ts[i] = self.lstm(time_step, (h_ts[i], c_ts[i]))
                outputs.append(torch.sigmoid(self.linear(h_ts[i])))
        # currently output is of shape time_steps x batch_size x output_size
        # convert it to batch_size x time_steps x output_size
        return outputs


class SkillDataSet(torch.utils.data.Dataset):
    def __init__(self, data, feature_depth, skill_depth):
        self.data = data
        self.user_ids = self.data['user_id'].unique()
        self.feature_depth = feature_depth
        self.skill_depth = skill_depth

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        # get the user_id at the given index
        user_id = self.user_ids[index]
        user_data = self.data[self.data['user_id'] == user_id]
        # order by order_id
        user_data = user_data.sort_values(by=['order_id'])
        # get input sequence
        input_sequence = torch.tensor(user_data['skill_with_answer'].to_numpy(), dtype=torch.float32)
        # One-hot encode the input sequence
        input_sequence = torch.nn.functional.one_hot(input_sequence.to(torch.int64), num_classes=self.feature_depth).to(torch.float32)        
        # One-hot encode the skill column
        output_sequence = torch.nn.functional.one_hot(torch.tensor(user_data['skill'].to_numpy(), dtype=torch.int64), num_classes=self.skill_depth).to(torch.float32)
        # get the label sequence
        label_sequence = torch.tensor(user_data['correct'].to_numpy(), dtype=torch.float32)
        # output sequence is currently of shape: (time_steps, num_skills)
        # Expand it to put the correct label at the end of each time step
        # So the shape becomes: (time_steps, num_skills + 1)
        # The last column is the label
        output_sequence = torch.cat((output_sequence, label_sequence.unsqueeze(1)), dim=1)


        #print(f'Input sequence shape: {input_sequence.shape} ')
        #print(f'Output sequence shape: {output_sequence.shape}')
        #print(f'_________')
        return input_sequence, output_sequence
    

class SkillDataSet2(torch.utils.data.Dataset):
    def __init__(self, data, feature_depth, skill_depth):
        self.data = data
        self.user_ids = self._get_filtered_user_ids()
        self.feature_depth = feature_depth
        self.skill_depth = skill_depth

    def __len__(self):
        return len(self.user_ids)
    
    def _get_filtered_user_ids(self):
        # Find the users that have at least 2 rows
        # This is because we need at least 2 rows to create a sequence
        user_ids = self.data['user_id'].unique()
        filtered_user_ids = []
        for user_id in user_ids:
            user_data = self.data[self.data['user_id'] == user_id]
            if len(user_data) > 1:
                filtered_user_ids.append(user_id)
        return filtered_user_ids
    
    def __getitem__(self, index):
        # get the user_id at the given index
        user_id = self.user_ids[index]
        user_data = self.data[self.data['user_id'] == user_id]
        # order by order_id
        user_data = user_data.sort_values(by=['order_id'])
        # get input sequence
        input_sequence = torch.tensor(user_data['skill_with_answer'].to_numpy(), dtype=torch.float32)
        # One-hot encode the input sequence
        input_sequence = torch.nn.functional.one_hot(input_sequence.to(torch.int64), num_classes=self.feature_depth).to(torch.float32)        
        # One-hot encode the skill column
        output_sequence = torch.nn.functional.one_hot(torch.tensor(user_data['skill'].to_numpy(), dtype=torch.int64), num_classes=self.skill_depth).to(torch.float32)
        # get the label sequence
        label_sequence = torch.tensor(user_data['correct'].to_numpy(), dtype=torch.float32)
        output_sequence = torch.cat((output_sequence, label_sequence.unsqueeze(1)), dim=1)
        # remove the last input_sequence
        input_sequence = input_sequence[:-1]
        # remove the first output_sequence
        output_sequence = output_sequence[1:]
        return input_sequence, output_sequence
    
