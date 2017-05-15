function apply_vad_apply_cmvn(inFile, vadFilename, outFile, rem_list)

%function list = apply_vad_apply_cmvn(inFile, vadFilename, outFile, rem_list)

% reads features with HTK format
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center
check = 0;
%disp('yolo')
vad=load(vadFilename);
%disp('yolo twice')
disp(size(vad))
if isempty(vad)
    check = 1;
    normdata = 0;
    return;
end

try
	[data,fp,dt,tc,t] = readhtk(inFile);
catch ME
	ME
	disp('readhtk porblem')
	%list = {rem_list, inFile};
%	disp('yolo thrice')
        return;
end
%if size(data,1) > size(data,2) 
  data = data';
%end
%data = data(1:40,:);
disp(size(data));
[D,N] = size(data) ;  
%if max(vad(end,2)) > N 
 %   disp('Warning : something wrong with VAD');
  %  vad(end,2) = N;
%end

vadBin = zeros(1,N);
disp(size(vad,2))
for I = 1:size(vad,1)
    vadBin(vad(I,1):vad(I,2)) = 1;
end


try	
	disp('disp vadbin')
	disp(vad(end,end))
	disp(size(vadBin))
        disp(N)
        disp(size(data,2))
	disp(size(data))
	data = data(:,find(vadBin == 1));
	
catch ME
	ME
%	list = rem_list;
        disp('yolo fourth')
	return
end
data=data';
writehtk(outFile,data,fp,tc);

%list = rem_list;
return;
