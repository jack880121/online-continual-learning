import torch


class Reservoir_update(object):
    def __init__(self, params):
        super().__init__()

    def update(self, buffer, x, y, **kwargs):
        batch_size = x.size(0)  #10

        # add whatever still fits in the buffer 用來先填滿buffer
        place_left = max(0, buffer.buffer_img.size(0) - buffer.current_index)   #size always 3000,current_index會遞增10 所以 place_left 遞減10

        if place_left:
            offset = min(place_left, batch_size)  #總是10，place_left遞減到最後會變0，不會進來
            buffer.buffer_img[buffer.current_index: buffer.current_index + offset].data.copy_(x[:offset])
            buffer.buffer_label[buffer.current_index: buffer.current_index + offset].data.copy_(y[:offset])


            buffer.current_index += offset     #初始0 累加10
            buffer.n_seen_so_far += offset     #初始0 累加10

        #TODO: the buffer tracker will have bug when the mem size can't be divided by batch size
        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]   #在place_left降到0之前，即是還沒填滿之前，這裡會把x,y清空變[]，底下幾樣也同樣是[]，place_left=0時x,y不改變
        
        # uniform_() tensor從均勻分布中抽樣數值進行填充，在buffer滿後隨機抽樣
        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, buffer.n_seen_so_far).long()
        #ex [29, 23, 14,  9, 10, 11, 22, 19,  3, 10]   buffer.buffer_img.size(0)=20
        valid_indices = (indices < buffer.buffer_img.size(0)).long()  #挑取有效的indices
        #ex [0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
        idx_new_data = valid_indices.nonzero().squeeze(-1)   #挑選出值不為0的index
        #ex [2, 3, 4, 5, 7, 8, 9]
        idx_buffer   = indices[idx_new_data]  #只取出有效的值
        #ex [14,  9, 10, 11, 19,  3, 10]
        buffer.n_seen_so_far += x.size(0)
        
        if idx_buffer.numel() == 0:     #numel()可以直接返回int類型的元素個數
            return []

        assert idx_buffer.max() < buffer.buffer_img.size(0)        #assert false時斷開
        assert idx_buffer.max() < buffer.buffer_label.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}

        # perform overwrite op   在buffer的[14,  9, 10, 11, 19,  3, 10]位置換上x的[2, 3, 4, 5, 7, 8, 9]位置
        buffer.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        buffer.buffer_label[list(idx_map.keys())] = y[list(idx_map.values())]
        return list(idx_map.keys())