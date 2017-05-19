%First doing it for overlap train. Done
%For single train. Done
%For single test. Done
%For overlap test
list = 'lists/fwrapvadoverlaptest.list';
[inFile, vadFilename, outFile] = textread(list,'%s %s %s');
removal_list = {};

for i = 1:length(inFile)
	i
	%list = apply_vad_apply_cmvn(inFile{i},vadFilename{i},outFile{i}, removal_list);
	vad_apply_kurtosis(inFile{i},vadFilename{i},outFile{i}, removal_list);
	%removal_list = {removal_list, list};
end
