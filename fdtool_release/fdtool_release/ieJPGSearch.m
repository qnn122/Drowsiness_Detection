function result = ieJPGSearch(searchString)
%% windows internet explorer + google search = URLs for jpg images
%
% inputs: simple search string for example 'navy sonar'
%
% outputs: first 100 URLs of jpg images
%
% example:
%    sonorURL = ieJPGSearch('navy sonar 2006');
%    disp(catURL)
%
% author - michaelB Nov 30, 2008
%
% rights - if you use this code, you must acknowledge michaelB
%          you can use any portion of this code anywhere
%
% updates: corrected lower case / upper case error

%% algorithm
if(~exist('searchString', 'var')),
    searchString = [];
end

if(isempty(searchString)),
    error('search string missing\n');
end

% open explorer
try
    wb=actxserver('internetexplorer.application');
 %   wb.set('visible', true);

catch,
    error('matlab failed to open windows internet explorer\n');
    return;

end

% change as you see fit
maxURLs = 100;
maxSearchIterations = 100;

% fix up search string for google address line
searchString = strrep(searchString, ' ', '+');

% the basic url for google image searches
googleSearchStringHead = 'http://images.google.com/images?&q=';

% concatenate
googleSearchString = [googleSearchStringHead, searchString];

allDone = false;

urlStart = 1;
nURL     = 0;
result   = cell(maxURLs, 1);
nIter    = 0;

while(~allDone)
    % where we start in goolge images
    urlStart = urlStart + nURL;

    % current search iteration
    nIter = nIter + 1;

    % enough results?
    if(nURL > maxURLs),
        result = {result{1:min(maxURLs, nURL)}}';
        return;
    end

    % too many iterations?
    if(nIter > maxSearchIterations),
        result = {result{1:min(maxURLs, nURL)}}';
        return;
    end

    % search string finally passed to google address line
    ss = [googleSearchString, sprintf('&start%d', urlStart)];

    % do the search
    wb.invoke('Navigate2', ss);

    % get the page HTML contents - must try a few times
    failed = true;
    maxFail = 100;
    nFail   = 0;
    pause(1);
    while(failed)
        try
            pause(0.05);
            pageInnerHTML = wb.document.documentElement.innerHTML;
            failed = false;
        catch,
            nFail = nFail + 1;
            if(nFail > maxFail),
                error('could not access page HTML code (timeout error)');
            end
        end
    end

    % very very very simple search for jpg images - search for the jpg file
    % extension - because of this, we might miss some files....
    jpgIndex = strfind(pageInnerHTML, '.jpg');

    % no jpg files found in HTML pages
    if(isempty(jpgIndex)),
        fprintf('\nno jpg files found\n');
        result = [];
        return;
    end

    % create temp space
    jpgUrl = cell(length(jpgIndex), 1);

    % iterate over jpg file names
    for k1=1:length(jpgIndex),
        try
            xLo = 1;
            if(k1 > 1), xLo = jpgIndex(k1-1); end
            xHi = jpgIndex(k1)+3;

            tempIndex = xLo:xHi;
            temp      = pageInnerHTML(tempIndex);

            httpIndex = strfind(temp, 'http');
            httpIndex = httpIndex(end);

            if(isempty(httpIndex)),
                continue;
            end

            jpgUrl{k1} = temp(httpIndex(1):end);
        end
    end

    % look for duplicate names and remove them
    for k1=1:length(jpgUrl),

        isDup = false;
        for k2=1:(k1-1),
            if(strcmp(lower(jpgUrl{k1}), lower(jpgUrl{k2})) == true),
                isDup = true;
                break;
            end
        end

        if(isDup == true), continue; end

        nURL = nURL + 1;
        result{nURL} = jpgUrl{k1};

    end
end

% ----------- michaelB Nov. 30 2008  -------------------


