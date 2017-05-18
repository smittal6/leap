
%inputFileList = '/home/leapery1/work/R149_0_1_wavs/lists/cmvn/mfcc_features_part01_part02.list';
%vadFilenameList = '/home/leapery1/work/R149_0_1_wavs/lists/cmvn/vad_part01_part02.list';
%outFileList = '/home/leapery1/work/R149_0_1_wavs/lists/cmvn/cmvn_features_part01_part02.list';
list = 'lists/fwrapvadoverlaptrain.list';
[inFile, vadFilename, outFile] = textread(list,'%s %s %s');
%inFile = textread(inputFileList, '%s');
%vadFilename = textread(vadFilenameList, '%s');
%outFile = textread(outFileList, '%s');

removal_list = {};

for i = 1:length(inFile)
	i
	%list = apply_vad_apply_cmvn(inFile{i},vadFilename{i},outFile{i}, removal_list);
	vad_apply_kurtosis(inFile{i},vadFilename{i},outFile{i}, removal_list);
	%removal_list = {removal_list, list};
end
