
list = '/home/neerajs/siddharth/lists/singletest/applyvadsingletest.list';
[inFile, vadFilename, outFile] = textread(list,'%s %s %s');
removal_list = {};
for i = 1:length(inFile)
	i
	apply_vad_apply_cmvn_1(inFile{i},vadFilename{i},outFile{i}, removal_list);
end
