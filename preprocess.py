import torch
import torch.nn as nn
import hgtk

hangul_set = ['']
hangul_set += list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ\"\' .?!,\n1234567890")


hangul_num = len(hangul_set)

def hangul_one_hot_tensor(string):
    tensor = torch.zeros(3 * len(string) * hangul_num).long()
    tensor = tensor.view(-1,3,len(hangul_set))
    
    for idx, char in enumerate(string):
        # 각각의 한글을 one-hot vector로 생성
        try:
            #자모 분리
            char_list = hgtk.letter.decompose(char)
            char_list = list(char_list)
            #분리 후
            tensor[idx][0][hangul_set.index(char_list[0])], tensor[idx][1][hangul_set.index(char_list[1])], tensor[idx][2][hangul_set.index(char_list[2])] = 1,1,1 #hangul_set.index(char_list[1]), hangul_set.index(char_list[2])

        except:
            #한글이 아닌 경우
            tensor[idx][0][hangul_set.index(char)] = 1
            pass
    
    return tensor

def hangul_int_tensor(string):
    tensor = torch.zeros(3 * len(string)).long()
    tensor = tensor.view(-1,3)
    
    for idx, char in enumerate(string):
        # 각각의 한글을 one-hot vector로 생성
        try:
            #자모 분리
            char_list = hgtk.letter.decompose(char)
            char_list = list(char_list)
            #분리 후
            tensor[idx][0], tensor[idx][1], tensor[idx][2] = hangul_set.index(char_list[0]), hangul_set.index(char_list[1]), hangul_set.index(char_list[2])

        except:
            #한글이 아닌 경우
            tensor[idx][0] = hangul_set.index(char)
            pass
    
    return tensor

if __name__=="__main__":
    print(hangul_one_hot_tensor("단발 머리를 나풀거리며 소녀가 막 달린다. 갈밭 사잇길로 들어섰다. 뒤에는 청량한 가을 햇살 아래 빛나는 갈꽃뿐."))
    print(f'size() : {hangul_one_hot_tensor("단발 머리를 나풀거리며 소녀가 막 달린다. 갈밭 사잇길로 들어섰다. 뒤에는 청량한 가을 햇살 아래 빛나는 갈꽃뿐.").size()}')
    
    print(hangul_int_tensor("단발 머리를 나풀거리며 소녀가 막 달린다. 갈밭 사잇길로 들어섰다. 뒤에는 청량한 가을 햇살 아래 빛나는 갈꽃뿐."))
    print(f'size() : {hangul_int_tensor("단발 머리를 나풀거리며 소녀가 막 달린다. 갈밭 사잇길로 들어섰다. 뒤에는 청량한 가을 햇살 아래 빛나는 갈꽃뿐.").size()}')
    

    embeddinglayer = nn.Embedding(hangul_num, 3)
    print(embeddinglayer.weight)
    print(f'size() : {embeddinglayer.weight.size()}')

    print(hangul_int_tensor("아아").size())
    print(embeddinglayer(hangul_int_tensor("아아")))

