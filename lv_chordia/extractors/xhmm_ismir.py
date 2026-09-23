"""
XHMMDecoder: turns per-frame chord probabilities into chord label segments.

chord_recognition.py builds an XHMMDecoder from a vocabulary -- a list of
chord names on root C (e.g. ``["C:maj", "C:min"]``, or the lines of one of
lv_chordia/data/*_chord_list.txt) that the decoder transposes to all twelve
roots -- then calls decode_to_chordlab() with the ensemble-averaged ChordNet
probabilities to get (start, end, chord_name) segments via Viterbi decoding.
An optional beat grid restricts chord changes to beats, with downbeats
cheapest (beat_frames()).

Reads: complex_chord.py
"""

import numpy as np
from ..complex_chord import shift_complex_chord_array,Chord,NUM_TO_ABS_SCALE


class XHMMDecoder():

    def __init__(self,chord_names,diff_trans_penalty=30.0,beat_trans_penalty=(15.0,45.0,100.0),
                 use_bass=True,use_7=True,use_extended=True):
        self.diff_trans_penalty=diff_trans_penalty
        self.beat_trans_penalty=beat_trans_penalty
        self.use_bass=use_bass
        self.use_7=use_7
        self.use_extended=use_extended
        self.__init_known_chord_names(chord_names)

    def __init_known_chord_names(self,chord_names):
        known_chord_array_pool={}
        for chord_name in chord_names:
            chord_name=chord_name.strip()
            if('/' in chord_name and not self.use_bass):
                continue
            if(':' in chord_name):
                tokens=chord_name.split(':')
                if(tokens[0]!='C'):
                    raise ValueError('vocabulary chord %r must be on root C (the decoder transposes it)'%chord_name)
                c=Chord(chord_name)
                array=c.to_numpy()
                if(-2 in array):
                    continue
                for shift in range(12):
                    shift_name='%s:%s'%(NUM_TO_ABS_SCALE[shift],tokens[1])
                    shift_array=tuple(shift_complex_chord_array(array,shift))
                    if(shift_array in known_chord_array_pool):
                        continue
                    known_chord_array_pool[shift_array]=shift_name
        self.known_chord_array=[((0,-1,-1,-1,-1,-1),'N')]+list(known_chord_array_pool.items())

    def get_chord_tag_obs(self,prob_list,triad_restriction=None):
        suffix_probs=[None]*4
        (prob_triad,prob_bass,suffix_probs[0],suffix_probs[1],suffix_probs[2],suffix_probs[3])=prob_list
        n_frame=prob_triad.shape[0]
        result_names=[]
        result_array=[]
        for (array,name) in self.known_chord_array:
            is_in_range=True
            for i in range(6):
                if(prob_list[i] is not None and array[i]>=prob_list[i].shape[-1]):
                    is_in_range=False
                    break
            if(is_in_range):
                assert(array[0]>=0)
                result_names.append(name)
                result_array.append(list(array))
        result_array=np.array(result_array,dtype=int)
        result_array[:,1]+=1 # bass adjust
        result_logprob=np.log(prob_triad[:,result_array[:,0]])
        bass_collect=result_array[:,1]>=0
        if(self.use_bass):
            result_logprob[:,bass_collect]+=np.log(prob_bass[:,result_array[bass_collect,1]])

        for i in range(4):
            if((i==0 and self.use_7) or (i>0 and self.use_extended)):
                suffix_collect=result_array[:,i+2]>=0
                roots=(result_array[:,0]-1)%12
                if(len(suffix_probs[i].shape)==3):
                    result_logprob[:,suffix_collect]+=\
                        np.log(suffix_probs[i][:,roots[suffix_collect],result_array[suffix_collect,i+2]])
                else:
                    result_logprob[:,suffix_collect]+=\
                        np.log(suffix_probs[i][:,result_array[suffix_collect,i+2]])

        if(triad_restriction is not None):
            triad_restriction=np.array(triad_restriction)
            result_logprob[result_array[None,:,0]!=triad_restriction[:,0,None]]=-np.inf
            result_logprob[result_array[None,:,1]!=triad_restriction[:,1,None]]=-np.inf

        return result_names,result_logprob

    def decode(self,prob_list,beat_arr,triad_restriction=None):
        result_names,result_logprob=self.get_chord_tag_obs(prob_list,triad_restriction)
        n_frame=result_logprob.shape[0]
        n_chord=result_logprob.shape[1]
        dp=np.zeros_like(result_logprob)
        dp[0,1:]-=np.inf
        dp_max_at=np.zeros((n_frame),dtype=int)
        pre=np.zeros_like(result_logprob,dtype=int)
        dp[0,:]+=result_logprob[0,:]
        dp_max_at[0]=np.argmax(dp[0,:])
        pre[0,:]=-1
        for t in range(1,n_frame):
            same_trans=dp[t-1,:]
            if(beat_arr[t]):
                diff_trans=dp[t-1,dp_max_at[t-1]]-(self.diff_trans_penalty if beat_arr[t]==1 else self.beat_trans_penalty[beat_arr[t]-2])
                use_same_trans=same_trans>diff_trans
                # dp[t-1,use_same_trans]=same_trans[use_same_trans]
                dp[t,:]=np.maximum(diff_trans,same_trans)+result_logprob[t,:]
                pre[t,:]=dp_max_at[t-1]
                pre[t,use_same_trans]=np.arange(n_chord)[use_same_trans]
            else:
                dp[t,:]=same_trans+result_logprob[t,:]
                pre[t,:]=np.arange(n_chord)
            dp_max_at[t]=np.argmax(dp[t,:])
        decode_ids=[None]*n_frame
        decode_ids[-1]=dp_max_at[-1]
        for t in range(n_frame-2,-1,-1):
            decode_ids[t]=pre[t+1,decode_ids[t+1]]
        return [result_names[i] for i in decode_ids]

    def decode_to_chordlab(self,prob_list,frame_seconds,beats=None,downbeats=False):
        """Decode to [start, end, name] segments; beats is an optional [(time, position-in-bar), ...] grid."""
        beat_arr=beat_frames(beats,prob_list[0].shape[0],frame_seconds,downbeats)
        decode_tags=self.decode(prob_list,beat_arr)
        result=[]
        last_frame=0
        n_frame=len(decode_tags)
        for i in range(n_frame):
            if(i+1==n_frame or decode_tags[i+1]!=decode_tags[i]):
                result.append([last_frame*frame_seconds,(i+1)*frame_seconds,decode_tags[i]])
                last_frame=i+1
        return result


def beat_frames(beats,length,frame_seconds,downbeats):
    """Per-frame change codes for decode(): 0 = no change allowed, 1 = free change (no grid),
    2 = downbeat, 3 = middle beat of an even bar, 4 = other beat (codes 2-4 only with downbeats).
    Frames before the first beat and after the last keep code 1."""
    beat_arr=np.ones((length,),dtype=np.int8)
    if(beats is not None):
        valid_beats=[(int(np.round(token[0]/frame_seconds)),int(np.round(token[1]))) for token in beats]
        valid_beats=[(token[0],token[1]) for token in valid_beats if token[0]>=0 and token[0]<beat_arr.shape[0]]
        for i in range(len(valid_beats)-1):
            beat_arr[valid_beats[i][0]+1:valid_beats[i+1][0]]=0
        if(downbeats and len(valid_beats)>0):
            num_beat_per_bar=np.max([token[1] for token in valid_beats])
            beat_arr[np.array([token[0] for token in valid_beats])]=4
            beat_arr[np.array([token[0] for token in valid_beats if token[1]==1])]=2
            if(num_beat_per_bar%2==0):
                beat_arr[np.array([token[0] for token in valid_beats if token[1]==num_beat_per_bar//2+1])]=3
    return beat_arr
